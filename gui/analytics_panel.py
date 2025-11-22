# gui/analytics_panel.py

from typing import Dict, List, Tuple
from collections import deque
from datetime import datetime, timedelta
import sys

from customtkinter import (
    CTkFrame,
    CTkLabel,
    CTkScrollableFrame,
    CTkProgressBar,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from .styles import Colors, Fonts, Sizes, create_card_frame, create_modern_button

def print_msg(message):
    """Simple print function to replace logging"""
    import datetime as dt
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - analytics_panel - INFO - {message}")

class AnalyticsPanel(CTkFrame):
    """Panel hiển thị analytics với charts và metrics thực tế"""
    
    def __init__(self, parent, state_manager, camera_manager, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.state = state_manager
        self.camera_manager = camera_manager
        
        # Data storage
        self.detection_history = deque(maxlen=1000)  # Lưu 1000 detections gần nhất
        self.alert_history = deque(maxlen=500)
        self.uptime_start = datetime.now()
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._create_header()
        self._create_content()
        
        # Auto refresh mỗi 5s
        self._start_auto_refresh()
    
    def _create_header(self):
        """Tạo header với time range selector"""
        header_frame = CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        CTkLabel(
            header_frame,
            text="📈 System Analytics",
            font=Fonts.TITLE_MD,
            text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        
        # Time range buttons
        btn_frame = CTkFrame(header_frame, fg_color="transparent")
        btn_frame.pack(side="right")
        
        for label, hours in [("1H", 1), ("6H", 6), ("24H", 24), ("7D", 168)]:
            create_modern_button(
                btn_frame,
                text=label,
                variant="secondary",
                size="small",
                command=lambda h=hours: self._update_time_range(h),
                width=60
            ).pack(side="left", padx=2)
    
    def _create_content(self):
        """Tạo content area"""
        container = CTkScrollableFrame(
            self,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.CORNER_RADIUS
        )
        container.grid(row=1, column=0, sticky="nsew", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        container.grid_columnconfigure((0, 1), weight=1)
        
        # Stats cards
        self._create_stats_section(container)
        
        # Charts
        self._create_charts_section(container)
        
        # Tables
        self._create_tables_section(container)
    
    def _create_stats_section(self, parent):
        """Tạo section với stat cards"""
        stats_frame = CTkFrame(parent, fg_color="transparent")
        stats_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, Sizes.PADDING_MD))
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Detection Rate
        self.detection_rate_card = self._create_metric_card(
            stats_frame, "Detection Rate", "0%", "↑ 0%", Colors.SUCCESS
        )
        self.detection_rate_card.grid(row=0, column=0, padx=Sizes.PADDING_SM, sticky="nsew")
        
        # False Positives
        self.false_positive_card = self._create_metric_card(
            stats_frame, "False Positives", "0%", "↓ 0%", Colors.SUCCESS
        )
        self.false_positive_card.grid(row=0, column=1, padx=Sizes.PADDING_SM, sticky="nsew")
        
        # Response Time
        self.response_time_card = self._create_metric_card(
            stats_frame, "Avg Response Time", "0s", "→ 0%", Colors.WARNING
        )
        self.response_time_card.grid(row=0, column=2, padx=Sizes.PADDING_SM, sticky="nsew")
        
        # Uptime
        self.uptime_card = self._create_metric_card(
            stats_frame, "System Uptime", "0h", "↑ 100%", Colors.SUCCESS
        )
        self.uptime_card.grid(row=0, column=3, padx=Sizes.PADDING_SM, sticky="nsew")
    
    def _create_metric_card(self, parent, title, value, change, color):
        """Tạo metric card"""
        card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        
        CTkLabel(
            card,
            text=title,
            font=Fonts.CAPTION,
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, 0))
        
        value_label = CTkLabel(
            card,
            text=value,
            font=Fonts.TITLE_LG,
            text_color=Colors.TEXT_PRIMARY
        )
        value_label.pack(anchor="w", padx=Sizes.PADDING_MD)
        
        change_label = CTkLabel(
            card,
            text=change,
            font=Fonts.SMALL,
            text_color=color
        )
        change_label.pack(anchor="w", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        # Store references for updates
        card.value_label = value_label
        card.change_label = change_label
        
        return card
    
    def _create_charts_section(self, parent):
        """Tạo section với charts"""
        # Detection Timeline Chart
        chart_card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        chart_card.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, Sizes.PADDING_MD))
        
        CTkLabel(
            chart_card,
            text="📊 Detection Timeline (Last 24 Hours)",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        # Create matplotlib chart
        self.timeline_fig = Figure(figsize=(12, 4), facecolor=Colors.BG_SECONDARY)
        self.timeline_ax = self.timeline_fig.add_subplot(111)
        self.timeline_canvas = FigureCanvasTkAgg(self.timeline_fig, chart_card)
        self.timeline_canvas.get_tk_widget().pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        self._update_timeline_chart()
        
        # Alert Distribution Chart
        dist_frame = CTkFrame(parent, fg_color="transparent")
        dist_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        dist_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Pie chart for alert types
        pie_card = create_card_frame(dist_frame, fg_color=Colors.BG_SECONDARY)
        pie_card.grid(row=0, column=0, padx=(0, Sizes.PADDING_SM), sticky="nsew")
        
        CTkLabel(
            pie_card,
            text="🎯 Alert Distribution",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        self.pie_fig = Figure(figsize=(6, 4), facecolor=Colors.BG_SECONDARY)
        self.pie_ax = self.pie_fig.add_subplot(111)
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, pie_card)
        self.pie_canvas.get_tk_widget().pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        self._update_pie_chart()
        
        # Bar chart for camera activity
        bar_card = create_card_frame(dist_frame, fg_color=Colors.BG_SECONDARY)
        bar_card.grid(row=0, column=1, padx=(Sizes.PADDING_SM, 0), sticky="nsew")
        
        CTkLabel(
            bar_card,
            text="📹 Camera Activity",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        self.bar_fig = Figure(figsize=(6, 4), facecolor=Colors.BG_SECONDARY)
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvasTkAgg(self.bar_fig, bar_card)
        self.bar_canvas.get_tk_widget().pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        self._update_bar_chart()
    
    def _create_tables_section(self, parent):
        """Tạo section với bảng dữ liệu"""
        table_card = create_card_frame(parent, fg_color=Colors.BG_SECONDARY)
        table_card.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(Sizes.PADDING_MD, 0))
        
        CTkLabel(
            table_card,
            text="📋 Recent Alerts",
            font=Fonts.HEADING,
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        
        # Table scroll frame
        self.table_scroll = CTkScrollableFrame(
            table_card,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.CORNER_RADIUS,
            height=200
        )
        self.table_scroll.pack(fill="both", expand=True, padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        self._update_alerts_table()
    
    def _update_timeline_chart(self):
        """Cập nhật timeline chart"""
        try:
            self.timeline_ax.clear()
            
            # Generate sample data (sẽ thay bằng dữ liệu thực)
            hours = list(range(24))
            detections = np.random.randint(0, 20, 24)
            
            self.timeline_ax.plot(hours, detections, color=Colors.PRIMARY, linewidth=2, marker='o')
            self.timeline_ax.fill_between(hours, detections, alpha=0.3, color=Colors.PRIMARY)
            
            self.timeline_ax.set_xlabel('Hours Ago', color=Colors.TEXT_SECONDARY)
            self.timeline_ax.set_ylabel('Detections', color=Colors.TEXT_SECONDARY)
            self.timeline_ax.set_facecolor(Colors.BG_PRIMARY)
            self.timeline_ax.tick_params(colors=Colors.TEXT_SECONDARY)
            self.timeline_ax.grid(True, alpha=0.2, color=Colors.BORDER)
            
            self.timeline_fig.tight_layout()
            self.timeline_canvas.draw()
        except Exception as e:
            print(f"Error updating timeline chart: {e}")
    
    def _update_pie_chart(self):
        """Cập nhật pie chart"""
        try:
            self.pie_ax.clear()
            
            # Sample data (sẽ thay bằng dữ liệu thực)
            labels = ['Known Person', 'Stranger', 'Fire Warning', 'Fire Critical']
            sizes = [45, 25, 20, 10]
            colors = [Colors.SUCCESS, Colors.WARNING, Colors.INFO, Colors.DANGER]
            
            self.pie_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, textprops={'color': Colors.TEXT_PRIMARY})
            self.pie_ax.set_facecolor(Colors.BG_SECONDARY)
            
            self.pie_fig.tight_layout()
            self.pie_canvas.draw()
        except Exception as e:
            print(f"Error updating pie chart: {e}")
    
    def _update_bar_chart(self):
        """Cập nhật bar chart"""
        try:
            self.bar_ax.clear()
            
            # Sample data (sẽ thay bằng dữ liệu thực từ camera_manager)
            cameras = list(self.camera_manager.cameras.keys()) if self.camera_manager else ['Cam 0']
            activity = np.random.randint(10, 100, len(cameras))
            
            bars = self.bar_ax.bar(range(len(cameras)), activity, color=Colors.PRIMARY)
            self.bar_ax.set_xticks(range(len(cameras)))
            self.bar_ax.set_xticklabels([f'Cam {i}' for i in range(len(cameras))], color=Colors.TEXT_SECONDARY)
            self.bar_ax.set_ylabel('Detections', color=Colors.TEXT_SECONDARY)
            self.bar_ax.set_facecolor(Colors.BG_PRIMARY)
            self.bar_ax.tick_params(colors=Colors.TEXT_SECONDARY)
            self.bar_ax.grid(True, alpha=0.2, color=Colors.BORDER, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                self.bar_ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', color=Colors.TEXT_PRIMARY)
            
            self.bar_fig.tight_layout()
            self.bar_canvas.draw()
        except Exception as e:
            print(f"Error updating bar chart: {e}")
    
    def _update_alerts_table(self):
        """Cập nhật bảng alerts"""
        try:
            # Clear existing
            for widget in self.table_scroll.winfo_children():
                widget.destroy()
            
            # Get recent alerts
            alerts = self.state.list_alerts()[-20:]  # 20 cái gần nhất
            
            if not alerts:
                CTkLabel(
                    self.table_scroll,
                    text="No recent alerts",
                    font=Fonts.BODY,
                    text_color=Colors.TEXT_MUTED
                ).pack(pady=Sizes.PADDING_LG)
                return
            
            # Create table rows
            for alert in reversed(alerts):
                row = CTkFrame(self.table_scroll, fg_color=Colors.BG_TERTIARY, corner_radius=Sizes.CORNER_RADIUS_SM)
                row.pack(fill="x", pady=2)
                row.grid_columnconfigure((0, 1, 2, 3), weight=1)
                
                # Time
                time_str = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                CTkLabel(row, text=time_str, font=Fonts.SMALL, text_color=Colors.TEXT_SECONDARY).grid(
                    row=0, column=0, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS, sticky="w"
                )
                
                # Type
                type_icon = "👤" if "person" in alert.type else "🔥"
                CTkLabel(row, text=f"{type_icon} {alert.type}", font=Fonts.SMALL, text_color=Colors.TEXT_PRIMARY).grid(
                    row=0, column=1, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS, sticky="w"
                )
                
                # Source
                CTkLabel(row, text=f"📹 {alert.source_id or 'N/A'}", font=Fonts.SMALL, text_color=Colors.TEXT_SECONDARY).grid(
                    row=0, column=2, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS, sticky="w"
                )
                
                # Status
                status_text = "✅ Resolved" if alert.resolved else "⏳ Pending"
                status_color = Colors.SUCCESS if alert.resolved else Colors.WARNING
                CTkLabel(row, text=status_text, font=Fonts.SMALL, text_color=status_color).grid(
                    row=0, column=3, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS, sticky="e"
                )
        except Exception as e:
            print(f"Error updating alerts table: {e}")
    
    def _update_metrics(self):
        """Cập nhật các metric cards"""
        try:
            # Detection Rate
            total_detections = len(self.detection_history)
            detection_rate = min(100, total_detections / 10)  # Simple calculation
            self.detection_rate_card.value_label.configure(text=f"{detection_rate:.1f}%")
            self.detection_rate_card.change_label.configure(text=f"↑ {detection_rate/10:.1f}%")
            
            # False Positives
            false_positive_rate = 3.5  # Placeholder
            self.false_positive_card.value_label.configure(text=f"{false_positive_rate:.1f}%")
            
            # Response Time
            avg_response = 1.2  # Placeholder
            self.response_time_card.value_label.configure(text=f"{avg_response:.1f}s")
            
            # Uptime
            uptime_delta = datetime.now() - self.uptime_start
            hours = uptime_delta.total_seconds() / 3600
            if hours < 1:
                uptime_text = f"{int(uptime_delta.total_seconds() / 60)}m"
            elif hours < 24:
                uptime_text = f"{hours:.1f}h"
            else:
                days = hours / 24
                uptime_text = f"{days:.1f}d"
            
            self.uptime_card.value_label.configure(text=uptime_text)
            uptime_pct = min(100, hours / 24 * 100)
            self.uptime_card.change_label.configure(text=f"↑ {uptime_pct:.1f}%")
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def _update_time_range(self, hours):
        """Cập nhật time range cho charts"""
        print(f"Time range updated to {hours} hours")
        # TODO: Filter data by time range
        self.refresh_data()
    
    def _start_auto_refresh(self):
        """Bắt đầu auto refresh"""
        def refresh_loop():
            try:
                self.refresh_data()
            except Exception as e:
                print(f"Auto refresh error: {e}")
            finally:
                self.after(5000, refresh_loop)  # Refresh mỗi 5s
        
        refresh_loop()
    
    def refresh_data(self):
        """Refresh tất cả dữ liệu"""
        try:
            self._update_metrics()
            self._update_timeline_chart()
            self._update_pie_chart()
            self._update_bar_chart()
            self._update_alerts_table()
        except Exception as e:
            print(f"Error refreshing analytics data: {e}")
    
    def log_detection(self, detection_type: str, source_id: str):
        """Log một detection event"""
        self.detection_history.append({
            'type': detection_type,
            'source': source_id,
            'timestamp': datetime.now()
        })
    
    def log_alert(self, alert_type: str, source_id: str, resolved: bool = False):
        """Log một alert event"""
        self.alert_history.append({
            'type': alert_type,
            'source': source_id,
            'resolved': resolved,
            'timestamp': datetime.now()
        })