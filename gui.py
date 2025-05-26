import sys
import vtk
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPixmap
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import open3d as o3d
import colorsys
import os


class VTKWidget(QVTKRenderWindowInteractor):
    def __init__(self, parent=None, sync_callback=None, window_id=1):
        super(VTKWidget, self).__init__(parent)
        self.Initialize()
        self.Start()
        self.sync_callback = sync_callback  # 同步回调函数
        self.window_id = window_id  # 添加窗口编号属性
        self.interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)
        self.renderer = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.renderer)
        self.actors = []
        self.normals_actors = []
        self.show_faces = True
        self.bounds_actor = None
        self.show_depth_map = False
        self.model = None
        self.filename = ""
        self.show_normals_flag = False  # 添加 show_normals_flag 属性
        self.selected_color = (0.5, 0.5, 0.5)  # 修改初始颜色为灰色 (RGB: 0.5, 0.5, 0.5)
        self.selected_size = 5  # 添加 selected_size 属性
        self.light = vtk.vtkLight()
        self.light.SetPosition(1, 1, 1)
        self.light.SetFocalPoint(0, 0, 0)
        self.renderer.AddLight(self.light)
        self.auto_rotate_timer = None
        self.rotation_angle = 0
        self.is_rotating = False
        self.renderer.SetBackground(1.0, 1.0, 1.0)

        # 添加窗口编号文本演员，修改为“窗口X”格式
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput(f"窗口{self.window_id}")  # 修改为“窗口X”格式
        self.text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)  # 黑色文字
        self.text_actor.GetTextProperty().SetFontSize(20)
        self.text_actor.GetTextProperty().SetFontFamilyToArial()
        self.text_actor.SetPosition(10, 10)  # 左上角位置
        self.renderer.AddActor2D(self.text_actor)

        # 绑定交互事件
        self.interactor = self.GetRenderWindow().GetInteractor()
        self.interactor.AddObserver("InteractionEvent", self.on_interaction)

    def on_interaction(self, obj, event):
        # 当用户交互时，获取当前相机的状态并通过回调同步到其他窗口
        camera = self.renderer.GetActiveCamera()
        if self.sync_callback:
            self.sync_callback(camera)

    def sync_camera(self, camera):
        # 同步相机的位置、焦点和视角
        self.renderer.GetActiveCamera().SetPosition(camera.GetPosition())
        self.renderer.GetActiveCamera().SetFocalPoint(camera.GetFocalPoint())
        self.renderer.GetActiveCamera().SetViewUp(camera.GetViewUp())
        self.renderer.GetActiveCamera().SetClippingRange(camera.GetClippingRange())
        self.GetRenderWindow().Render()

    def show_cloud(self, cloud, filename, color=(0.5, 0.5, 0.5), point_size=5, show_normals=False, use_depth_map=False):
        self.model = cloud
        self.filename = filename
        self.show_normals_flag = show_normals  # 更新 show_normals_flag
        self.selected_color = color  # 更新 selected_color
        self.selected_size = point_size  # 更新 selected_size

        polydata = vtk.vtkPolyData()
        if isinstance(cloud, o3d.geometry.TriangleMesh):
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(np.asarray(cloud.vertices)))
            polydata.SetPoints(points)

            if cloud.has_triangles():
                cells = vtk.vtkCellArray()
                triangles = np.asarray(cloud.triangles)
                for tri in triangles:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, tri[0])
                    triangle.GetPointIds().SetId(1, tri[1])
                    triangle.GetPointIds().SetId(2, tri[2])
                    cells.InsertNextCell(triangle)
                polydata.SetPolys(cells)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                if cloud.has_triangle_uvs() and cloud.textures and len(cloud.textures) > 0:
                    uvs = np.asarray(cloud.triangle_uvs)
                    tcoords = vtk.vtkFloatArray()
                    tcoords.SetNumberOfComponents(2)
                    tcoords.SetName("TextureCoordinates")
                    for uv in uvs:
                        tcoords.InsertNextTuple2(uv[0], uv[1])
                    polydata.GetPointData().SetTCoords(tcoords)

                    texture = vtk.vtkTexture()
                    texture_file = cloud.textures[0]
                    if not os.path.exists(texture_file):
                        print(f"Warning: Texture file {texture_file} not found!")
                    else:
                        if texture_file.endswith('.png'):
                            texture_image = vtk.vtkPNGReader()
                        else:
                            texture_image = vtk.vtkJPEGReader()
                        texture_image.SetFileName(texture_file)
                        texture_image.Update()
                        texture.SetInputConnection(texture_image.GetOutputPort())
                        texture.InterpolateOn()
                        actor.SetTexture(texture)
                        print(f"Applied texture from {texture_file}")

                elif cloud.has_vertex_colors():
                    colors = vtk.vtkUnsignedCharArray()
                    colors.SetNumberOfComponents(3)
                    colors.SetName("Colors")
                    vertex_colors = np.asarray(cloud.vertex_colors) * 255
                    for rgb in vertex_colors:
                        colors.InsertNextTuple3(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                    polydata.GetPointData().SetScalars(colors)
                    mapper.SetScalarVisibility(1)
                    print("Applied vertex colors")

                elif use_depth_map:
                    self._set_depth_map_color(np.asarray(cloud.vertices), mode='rainbow', polydata=polydata, actor=actor)
                else:
                    actor.GetProperty().SetColor(self.selected_color)  # 使用 gray color

                if self.show_faces:
                    actor.GetProperty().SetRepresentationToSurface()
                    actor.GetProperty().SetLighting(True)
                else:
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetLighting(False)
        else:
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(np.asarray(cloud.points)))
            polydata.SetPoints(points)
            vertex_filter = vtk.vtkVertexGlyphFilter()
            vertex_filter.SetInputData(polydata)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex_filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            if cloud.has_colors():
                colors = vtk.vtkUnsignedCharArray()
                colors.SetNumberOfComponents(3)
                colors.SetName("Colors")
                point_colors = np.asarray(cloud.colors) * 255
                for rgb in point_colors:
                    colors.InsertNextTuple3(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                polydata.GetPointData().SetScalars(colors)
                mapper.SetScalarVisibility(1)
            elif use_depth_map:
                self._set_depth_map_color(np.asarray(cloud.points), mode='rainbow', polydata=polydata, actor=actor)
            else:
                actor.GetProperty().SetPointSize(point_size)
                actor.GetProperty().SetColor(self.selected_color)  # 使用 gray color

        self.actors.append(actor)
        self.renderer.AddActor(actor)
        if self.show_normals_flag:
            self.show_normals(cloud)

        self.renderer.ResetCamera()
        self.GetRenderWindow().Render()

    def _set_depth_map_color(self, points, mode='rainbow', polydata=None, actor=None):
        z_values = points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        if z_max == z_min:
            z_max = z_min + 1e-6

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        for z in z_values:
            z_normalized = (z - z_min) / (z_max - z_min)
            if mode == 'rainbow':
                hue = 240 * (1 - z_normalized) / 360
                saturation = 1.0
                value = 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            else:
                gray = int(255 * z_normalized)
                r, g, b = gray, gray, gray
            colors.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))

        polydata.GetPointData().SetScalars(colors)
        actor.GetMapper().SetScalarVisibility(1)

    def show_normals(self, cloud):
        if isinstance(cloud, o3d.geometry.TriangleMesh):
            if not cloud.has_vertex_normals():
                cloud.compute_vertex_normals()
            normals = np.asarray(cloud.vertex_normals)
            points = np.asarray(cloud.vertices)
        else:
            if not cloud.has_normals():
                return
            normals = np.asarray(cloud.normals)
            points = np.asarray(cloud.points)

        lines = vtk.vtkCellArray()
        points_vtk = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        num_points = len(points)
        z_values = normals[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)
        if z_max == z_min:
            z_max = z_min + 1e-6

        for i in range(num_points):
            z_normalized = (z_values[i] - z_min) / (z_max - z_min)
            hue = 240 * (1 - z_normalized) / 360
            saturation = 1.0
            value = 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))
            points_vtk.InsertNextPoint(points[i])
            points_vtk.InsertNextPoint(points[i] + normals[i] * 0.1)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(2 * i)
            lines.InsertCellPoint(2 * i + 1)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points_vtk)
        polydata.SetLines(lines)
        polydata.GetPointData().SetScalars(colors)

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputData(polydata)

        normals_actor = vtk.vtkActor()
        normals_actor.SetMapper(line_mapper)
        normals_actor.GetProperty().SetLineWidth(2)

        self.normals_actors.append(normals_actor)
        self.renderer.AddActor(normals_actor)

    def toggle_faces(self):
        self.show_faces = not self.show_faces
        for actor in self.actors:
            if actor.GetProperty().GetRepresentation() in [vtk.VTK_SURFACE, vtk.VTK_WIREFRAME]:
                if self.show_faces:
                    actor.GetProperty().SetRepresentationToSurface()
                    actor.GetProperty().SetLighting(True)
                else:
                    actor.GetProperty().SetRepresentationToWireframe()
                    actor.GetProperty().SetLighting(False)
        self.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        self.GetRenderWindow().Render()
        self.stop_auto_rotation()
        if self.sync_callback:
            self.sync_callback(camera)

    def set_background_color(self, color):
        self.renderer.SetBackground(color[0], color[1], color[2])
        self.GetRenderWindow().Render()

    def show_bounds(self, show_bounds):
        if show_bounds:
            if not self.bounds_actor and self.actors:
                combined_bounds = None
                for actor in self.actors:
                    bounds = actor.GetBounds()
                    if combined_bounds is None:
                        combined_bounds = list(bounds)
                    else:
                        combined_bounds[0] = min(combined_bounds[0], bounds[0])
                        combined_bounds[1] = max(combined_bounds[1], bounds[1])
                        combined_bounds[2] = min(combined_bounds[2], bounds[2])
                        combined_bounds[3] = max(combined_bounds[3], bounds[3])
                        combined_bounds[4] = min(combined_bounds[4], bounds[4])
                        combined_bounds[5] = max(combined_bounds[5], bounds[5])
                if combined_bounds[0] == combined_bounds[1]:
                    combined_bounds = [0, 1, 0, 1, 0, 1]
                cube = vtk.vtkCubeSource()
                cube.SetBounds(combined_bounds)
                cube_mapper = vtk.vtkPolyDataMapper()
                cube_mapper.SetInputConnection(cube.GetOutputPort())
                self.bounds_actor = vtk.vtkActor()
                self.bounds_actor.SetMapper(cube_mapper)
                self.bounds_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
                self.bounds_actor.GetProperty().SetLineWidth(2)
                self.bounds_actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(self.bounds_actor)
        else:
            if self.bounds_actor:
                self.renderer.RemoveActor(self.bounds_actor)
                self.bounds_actor = None
        self.GetRenderWindow().Render()

    def toggle_depth_map(self, use_depth_map):
        self.show_depth_map = use_depth_map
        if self.model:
            self.clear_actors()
            self.show_cloud(self.model, self.filename, color=self.selected_color, point_size=self.selected_size,
                           show_normals=self.show_normals_flag, use_depth_map=self.show_depth_map)

    def clear_actors(self):
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        for normals_actor in self.normals_actors:
            self.renderer.RemoveActor(normals_actor)
        self.actors = []
        self.normals_actors = []

    def save_model(self, file_path):
        if self.model:
            if isinstance(self.model, o3d.geometry.PointCloud):
                if file_path.endswith('.pcd'):
                    o3d.io.write_point_cloud(file_path, self.model)
                else:
                    o3d.io.write_point_cloud(f"{file_path}.pcd", self.model)
            elif isinstance(self.model, o3d.geometry.TriangleMesh):
                if file_path.endswith('.ply'):
                    o3d.io.write_triangle_mesh(file_path, self.model)
                else:
                    o3d.io.write_triangle_mesh(f"{file_path}.ply", self.model)
            print(f"模型已保存至：{file_path}")

    def save_screenshot(self, file_path):
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.GetRenderWindow())
        window_to_image.Update()

        writer = vtk.vtkPNGWriter()
        if not file_path.endswith('.png'):
            file_path += '.png'
        writer.SetFileName(file_path)
        writer.SetInputConnection(window_to_image.GetOutputPort())
        writer.Write()
        print(f"截图已保存至：{file_path}")

    def adjust_light_direction(self, x, y, z):
        self.light.SetPosition(x, y, z)
        self.light.SetFocalPoint(0, 0, 0)
        self.GetRenderWindow().Render()

    def start_auto_rotation(self):
        if not self.is_rotating and self.actors:
            self.is_rotating = True
            if not self.auto_rotate_timer:
                self.auto_rotate_timer = QTimer(self)
                self.auto_rotate_timer.timeout.connect(self.rotate_model)
                self.auto_rotate_timer.start(50)

    def stop_auto_rotation(self):
        if self.is_rotating:
            self.is_rotating = False
            if self.auto_rotate_timer:
                self.auto_rotate_timer.stop()
                self.auto_rotate_timer = None

    def rotate_model(self):
        if self.is_rotating and self.actors:
            self.rotation_angle += 2
            if self.rotation_angle >= 360:
                self.rotation_angle = 0
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(2)
            self.GetRenderWindow().Render()
            if self.sync_callback:
                self.sync_callback(camera)


class DemoWidget(QWidget):
    def __init__(self, parent=None):
        super(DemoWidget, self).__init__(parent)
        self.setWindowTitle("室外无约束场景的三维重建与渲染平台")
        
        # 创建主水平布局
        main_layout = QHBoxLayout()

        # 左侧：垂直布局，包含模型和统计信息
        self.left_layout = QVBoxLayout()
        
        # 左侧上方：动态模型窗口布局（改为 QVBoxLayout）
        self.model_layout = QVBoxLayout()
        self.vtk_widgets = []  # 存储多个 VTKWidget
        self.add_vtk_widget()  # 初始添加一个窗口
        self.left_layout.addLayout(self.model_layout, stretch=3)

        # 左侧下方：统计信息和图像显示区域
        bottom_left_layout = QVBoxLayout()
        
        # 统计信息
        stats_label = QLabel("统计信息：无模型加载")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_label.setStyleSheet("background-color: #808080; padding: 5px; font-size: 12px;")
        bottom_left_layout.addWidget(stats_label)
        self.stats_label = stats_label

        # 添加图像显示区域
        self.image_layout = QHBoxLayout()
        
        # 第一个图像框和“原图”标签
        image1_container = QVBoxLayout()
        image1_label = QLabel("原图")
        image1_label.setAlignment(Qt.AlignCenter)
        self.image_label1 = QLabel("图像1未加载")
        self.image_label1.setAlignment(Qt.AlignCenter)
        self.image_label1.setStyleSheet("background-color: #D3D3D3; border: 1px solid #808080;")
        self.image_label1.setScaledContents(True)
        image1_container.addWidget(image1_label)
        image1_container.addWidget(self.image_label1)
        
        # 第二个图像框和“渲染图像”标签
        image2_container = QVBoxLayout()
        image2_label = QLabel("渲染图像")
        image2_label.setAlignment(Qt.AlignCenter)
        self.image_label2 = QLabel("图像2未加载")
        self.image_label2.setAlignment(Qt.AlignCenter)
        self.image_label2.setStyleSheet("background-color: #D3D3D3; border: 1px solid #808080;")
        self.image_label2.setScaledContents(True)
        image2_container.addWidget(image2_label)
        image2_container.addWidget(self.image_label2)
        
        self.image_layout.addLayout(image1_container)
        self.image_layout.addLayout(image2_container)
        bottom_left_layout.addLayout(self.image_layout)
        
        self.left_layout.addLayout(bottom_left_layout)
        main_layout.addLayout(self.left_layout, stretch=3)

        # 右侧：垂直排列按钮和滑块
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        # 创建UI组件
        self.pushButton = QPushButton(self)
        self.pushButton.setText("加载三维模型")
        self.pushButton.clicked.connect(self.show_cloud)
        right_layout.addWidget(self.pushButton)

        self.customColorButton = QPushButton(self)
        self.customColorButton.setText("更改三维模型颜色")
        self.customColorButton.clicked.connect(self.select_custom_color)
        right_layout.addWidget(self.customColorButton)

        self.showNormalsButton = QPushButton(self)
        self.showNormalsButton.setText("显示/隐藏法线")
        self.showNormalsButton.clicked.connect(self.toggle_normals)
        right_layout.addWidget(self.showNormalsButton)

        self.showFacesButton = QPushButton(self)
        self.showFacesButton.setText("显示/隐藏面")
        self.showFacesButton.clicked.connect(self.toggle_faces)
        right_layout.addWidget(self.showFacesButton)

        self.resetButton = QPushButton(self)
        self.resetButton.setText("复位")
        self.resetButton.clicked.connect(self.reset_model)
        right_layout.addWidget(self.resetButton)

        self.backgroundButton = QPushButton(self)
        self.backgroundButton.setText("更改背景颜色")
        self.backgroundButton.clicked.connect(self.select_background_color)
        right_layout.addWidget(self.backgroundButton)

        self.boundsButton = QPushButton(self)
        self.boundsButton.setText("显示/隐藏边界框")
        self.boundsButton.clicked.connect(self.toggle_bounds)
        right_layout.addWidget(self.boundsButton)

        self.depthMapButton = QPushButton(self)
        self.depthMapButton.setText("显示/隐藏深度")
        self.depthMapButton.clicked.connect(self.toggle_depth_map)
        right_layout.addWidget(self.depthMapButton)

        self.saveButton = QPushButton(self)
        self.saveButton.setText("保存")
        self.saveButton.clicked.connect(self.save_model_and_view)
        right_layout.addWidget(self.saveButton)

        self.screenshotButton = QPushButton(self)
        self.screenshotButton.setText("截图")
        self.screenshotButton.clicked.connect(self.take_screenshot)
        right_layout.addWidget(self.screenshotButton)

        self.autoRotateButton = QPushButton(self)
        self.autoRotateButton.setText("自动旋转")
        self.autoRotateButton.clicked.connect(self.toggle_auto_rotation)
        right_layout.addWidget(self.autoRotateButton)

        self.renderImageButton = QPushButton(self)
        self.renderImageButton.setText("渲染图像")
        self.renderImageButton.clicked.connect(self.render_images)
        right_layout.addWidget(self.renderImageButton)

        # 添加光照方向调整滑块和标签
        right_layout.addWidget(QLabel("光照方向调整"))
        self.lightXSlider = QSlider(Qt.Horizontal, self)
        self.lightXSlider.setRange(-10, 10)
        self.lightXSlider.setValue(1)
        self.lightXSlider.valueChanged.connect(self.update_light_direction)
        right_layout.addWidget(QLabel("X 方向"))
        right_layout.addWidget(self.lightXSlider)

        self.lightYSlider = QSlider(Qt.Horizontal, self)
        self.lightYSlider.setRange(-10, 10)
        self.lightYSlider.setValue(1)
        self.lightYSlider.valueChanged.connect(self.update_light_direction)
        right_layout.addWidget(QLabel("Y 方向"))
        right_layout.addWidget(self.lightYSlider)

        self.lightZSlider = QSlider(Qt.Horizontal, self)
        self.lightZSlider.setRange(-10, 10)
        self.lightZSlider.setValue(1)
        self.lightZSlider.valueChanged.connect(self.update_light_direction)
        right_layout.addWidget(QLabel("Z 方向"))
        right_layout.addWidget(self.lightZSlider)

        self.lightDirectionLabel = QLabel("光照方向：(1, 1, 1)")
        self.lightDirectionLabel.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lightDirectionLabel)

        # 添加滑块和标签
        right_layout.addWidget(QLabel("点云大小（仅点模式有效）"))
        self.sizeSlider = QSlider(Qt.Horizontal, self)
        self.sizeSlider.setRange(1, 20)
        self.sizeSlider.setValue(5)
        self.sizeSlider.valueChanged.connect(self.update_point_size)
        right_layout.addWidget(self.sizeSlider)

        right_layout.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setStyleSheet("background-color: #D3D3D3; padding: 5px;")
        main_layout.addWidget(right_widget, stretch=1)

        self.setLayout(main_layout)
        
        self.show_depth_map = False  # 仅保留必要的全局属性
        self.loaded_models = 0  # 记录加载的模型数量

    def sync_camera(self, camera):
        # 同步所有窗口的相机
        for vtk_widget in self.vtk_widgets:
            if vtk_widget.model:
                vtk_widget.sync_camera(camera)

    def add_vtk_widget(self):
        # 窗口编号从 1 开始，递增
        window_id = len(self.vtk_widgets) + 1
        vtk_widget = VTKWidget(sync_callback=self.sync_camera, window_id=window_id)
        self.vtk_widgets.append(vtk_widget)

    def update_model_layout(self):
        # 清空当前布局中的所有 widget
        for i in reversed(range(self.model_layout.count())):
            item = self.model_layout.itemAt(i)
            if item.layout():
                layout = item.layout()
                for j in reversed(range(layout.count())):
                    widget = layout.itemAt(j).widget()
                    if widget:
                        layout.removeWidget(widget)
                        widget.setParent(None)
                self.model_layout.removeItem(item)
            elif item.widget():
                widget = item.widget()
                self.model_layout.removeWidget(widget)
                widget.setParent(None)

        # 根据当前模型数量重新分配窗口
        num_models = self.loaded_models
        if num_models == 0:
            return

        # 创建布局
        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()

        if num_models == 1:
            # 1 个模型：第一行 1 个窗口
            row1_layout.addWidget(self.vtk_widgets[0])
        elif num_models == 2:
            # 2 个模型：上下两个窗口
            row1_layout.addWidget(self.vtk_widgets[0])
            row2_layout.addWidget(self.vtk_widgets[1])
        elif num_models == 3:
            # 3 个模型：第一行 2 个，第二行 1 个
            row1_layout.addWidget(self.vtk_widgets[0])
            row1_layout.addWidget(self.vtk_widgets[1])
            row2_layout.addWidget(self.vtk_widgets[2])
        elif num_models == 4:
            # 4 个模型：上下分别两个窗口
            row1_layout.addWidget(self.vtk_widgets[0])
            row1_layout.addWidget(self.vtk_widgets[1])
            row2_layout.addWidget(self.vtk_widgets[2])
            row2_layout.addWidget(self.vtk_widgets[3])
        elif num_models == 5:
            # 5 个模型：第一行 3 个，第二行 2 个
            row1_layout.addWidget(self.vtk_widgets[0])
            row1_layout.addWidget(self.vtk_widgets[1])
            row1_layout.addWidget(self.vtk_widgets[2])
            row2_layout.addWidget(self.vtk_widgets[3])
            row2_layout.addWidget(self.vtk_widgets[4])
        elif num_models == 6:
            # 6 个模型：第一行 3 个，第二行 3 个
            row1_layout.addWidget(self.vtk_widgets[0])
            row1_layout.addWidget(self.vtk_widgets[1])
            row1_layout.addWidget(self.vtk_widgets[2])
            row2_layout.addWidget(self.vtk_widgets[3])
            row2_layout.addWidget(self.vtk_widgets[4])
            row2_layout.addWidget(self.vtk_widgets[5])

        # 将布局添加到主布局
        self.model_layout.addLayout(row1_layout)
        if num_models >= 2:
            self.model_layout.addLayout(row2_layout)

        # 重新加载模型
        for i in range(min(self.loaded_models, 6)):  # 最多显示 6 个模型
            vtk_widget = self.vtk_widgets[i]
            if vtk_widget.model:
                vtk_widget.show_cloud(vtk_widget.model, vtk_widget.filename, color=vtk_widget.selected_color,
                                     point_size=vtk_widget.selected_size, show_normals=vtk_widget.show_normals_flag,
                                     use_depth_map=vtk_widget.show_depth_map)

    def show_cloud(self):
        fileName, filetype = QFileDialog.getOpenFileName(self, "请选择点云：", '.', "Cloud Files (*.pcd *.ply)")
        if fileName != '':
            try:
                if fileName.endswith('.ply'):
                    pcd = o3d.io.read_triangle_mesh(fileName)
                    if not pcd.has_vertices():
                        pcd = o3d.io.read_point_cloud(fileName)
                    # 处理纹理
                    if pcd.has_triangle_uvs() and not pcd.textures:
                        texture_file_guess = os.path.splitext(fileName)[0] + '.png'
                        if os.path.exists(texture_file_guess):
                            pcd.textures = [texture_file_guess]
                        else:
                            texture_file_guess = os.path.splitext(fileName)[0] + '.jpg'
                            if os.path.exists(texture_file_guess):
                                pcd.textures = [texture_file_guess]
                            else:
                                texture_file, _ = QFileDialog.getOpenFileName(self, "请选择纹理图像", '.', "Image Files (*.png *.jpg *.jpeg)")
                                if texture_file:
                                    pcd.textures = [texture_file]
                else:
                    pcd = o3d.io.read_point_cloud(fileName)

                if not (pcd.has_vertices() if isinstance(pcd, o3d.geometry.TriangleMesh) else pcd.has_points()):
                    QMessageBox.warning(self, "错误", "加载的文件为空！")
                    return
                
                # 初始法线显示状态
                initial_show_normals = fileName.endswith('.ply') and (pcd.has_vertex_normals() if isinstance(pcd, o3d.geometry.TriangleMesh) else pcd.has_normals())

                # 分配模型到窗口
                self.loaded_models += 1
                if self.loaded_models > len(self.vtk_widgets):
                    self.add_vtk_widget()
                if self.loaded_models > 6:  # 限制最多加载 6 个模型
                    self.loaded_models = 6
                    QMessageBox.warning(self, "警告", "最多支持加载 6 个模型！")
                    return
                self.vtk_widgets[self.loaded_models - 1].show_cloud(pcd, fileName.split('/')[-1], color=(0.5, 0.5, 0.5),  # 初始颜色为灰色
                                                                  point_size=5, show_normals=initial_show_normals,
                                                                  use_depth_map=self.show_depth_map)
                self.update_model_layout()

                # 更新统计信息（只显示最后一个加载的模型）
                stats_text = "统计信息：\n"
                vtk_widget = self.vtk_widgets[self.loaded_models - 1]
                if vtk_widget.model:
                    filename = vtk_widget.filename
                    model = vtk_widget.model
                    if isinstance(model, o3d.geometry.PointCloud):
                        vertex_count = len(model.points)
                        stats_text += f"Model: {filename}\nVertices: {vertex_count:,}\nFaces: 0\n"
                    elif isinstance(model, o3d.geometry.TriangleMesh):
                        vertex_count = len(model.vertices)
                        face_count = len(model.triangles) if model.has_triangles() else 0
                        stats_text += f"Model: {filename}\nVertices: {vertex_count:,}\nFaces: {face_count:,}\n"
                self.stats_label.setText(stats_text.strip() if stats_text.strip() else "统计信息：无模型加载")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
    
    def select_custom_color(self):
        color = QColorDialog.getColor(initial=QColor.fromRgbF(*self.vtk_widgets[0].selected_color if self.vtk_widgets else (0.5, 0.5, 0.5)), parent=self, title="选择颜色")
        if color.isValid():
            new_color = (color.redF(), color.greenF(), color.blueF())
            # 提示用户选择要更改颜色的窗口
            items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
            if not items:
                QMessageBox.warning(self, "警告", "没有加载模型！")
                return
            item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要更改颜色的窗口：", items, 0, False)
            if ok and item:
                idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
                vtk_widget = self.vtk_widgets[idx]
                if vtk_widget.model:
                    vtk_widget.selected_color = new_color
                    vtk_widget.clear_actors()
                    vtk_widget.show_cloud(vtk_widget.model, vtk_widget.filename, color=vtk_widget.selected_color,
                                        point_size=vtk_widget.selected_size, show_normals=vtk_widget.show_normals_flag,
                                        use_depth_map=vtk_widget.show_depth_map)

    def toggle_normals(self):
        # 提示用户选择要切换法线的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要切换法线的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.show_normals_flag = not vtk_widget.show_normals_flag
            if vtk_widget.model:
                vtk_widget.clear_actors()
                vtk_widget.show_cloud(vtk_widget.model, vtk_widget.filename, color=vtk_widget.selected_color,
                                    point_size=vtk_widget.selected_size, show_normals=vtk_widget.show_normals_flag,
                                    use_depth_map=vtk_widget.show_depth_map)

    def toggle_faces(self):
        # 提示用户选择要切换面显示的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要切换面显示的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.toggle_faces()

    def reset_model(self):
        for vtk_widget in self.vtk_widgets:
            vtk_widget.reset_camera()
            vtk_widget.stop_auto_rotation()

    def update_point_size(self, value):
        # 提示用户选择要调整点云大小的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要调整点云大小的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.selected_size = value
            if vtk_widget.model:
                vtk_widget.clear_actors()
                vtk_widget.show_cloud(vtk_widget.model, vtk_widget.filename, color=vtk_widget.selected_color,
                                    point_size=vtk_widget.selected_size, show_normals=vtk_widget.show_normals_flag,
                                    use_depth_map=vtk_widget.show_depth_map)

    def update_light_direction(self, value):
        x = self.lightXSlider.value()
        y = self.lightYSlider.value()
        z = self.lightZSlider.value()
        # 提示用户选择要调整光照的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要调整光照的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.adjust_light_direction(x, y, z)
        self.lightDirectionLabel.setText(f"光照方向：({x}, {y}, {z})")

    def select_background_color(self):
        color = QColorDialog.getColor(initial=QColor.fromRgbF(1.0, 1.0, 1.0), parent=self, title="选择背景颜色")
        if color.isValid():
            bg_color = (color.redF(), color.greenF(), color.blueF())
            # 提示用户选择要更改背景颜色的窗口
            items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
            if not items:
                QMessageBox.warning(self, "警告", "没有加载模型！")
                return
            item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要更改背景颜色的窗口：", items, 0, False)
            if ok and item:
                idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
                vtk_widget = self.vtk_widgets[idx]
                vtk_widget.set_background_color(bg_color)

    def toggle_bounds(self):
        # 提示用户选择要切换边界框的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要切换边界框的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.show_bounds(not bool(vtk_widget.bounds_actor))

    def toggle_depth_map(self):
        # 提示用户选择要切换深度图的窗口
        items = [f"窗口{i}" for i in range(1, self.loaded_models + 1)]  # 修改为“窗口X”格式
        if not items:
            QMessageBox.warning(self, "警告", "没有加载模型！")
            return
        item, ok = QInputDialog.getItem(self, "选择窗口", "请选择要切换深度图的窗口：", items, 0, False)
        if ok and item:
            idx = int(item.replace("窗口", "")) - 1  # 从“窗口X”提取编号并转换为索引
            vtk_widget = self.vtk_widgets[idx]
            vtk_widget.show_depth_map = not vtk_widget.show_depth_map
            if vtk_widget.model:
                vtk_widget.clear_actors()
                vtk_widget.show_cloud(vtk_widget.model, vtk_widget.filename, color=vtk_widget.selected_color,
                                    point_size=vtk_widget.selected_size, show_normals=vtk_widget.show_normals_flag,
                                    use_depth_map=vtk_widget.show_depth_map)

    def save_model_and_view(self):
        options = QMessageBox.question(self, "保存选项", "您想保存模型还是视图？",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        
        if options == QMessageBox.StandardButton.Yes:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存模型", ".", "Model Files (*.ply *.pcd)")
            if file_name:
                for i, vtk_widget in enumerate(self.vtk_widgets):
                    if vtk_widget.model:
                        suffix = f"_{i+1}" if self.loaded_models > 1 else ""  # 修改后缀为窗口编号
                        vtk_widget.save_model(f"{file_name}{suffix}")
        elif options == QMessageBox.StandardButton.No:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存视图", ".", "Image Files (*.png)")
            if file_name:
                for i, vtk_widget in enumerate(self.vtk_widgets):
                    if vtk_widget.model:
                        suffix = f"_{i+1}" if self.loaded_models > 1 else ""  # 修改后缀为窗口编号
                        vtk_widget.save_screenshot(f"{file_name}{suffix}")

    def take_screenshot(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "保存截图", ".", "Image Files (*.png)")
        if file_name:
            for i, vtk_widget in enumerate(self.vtk_widgets):
                if vtk_widget.model:
                    suffix = f"_{i+1}" if self.loaded_models > 1 else ""  # 修改后缀为窗口编号
                    vtk_widget.save_screenshot(f"{file_name}{suffix}")

    def toggle_auto_rotation(self):
        for vtk_widget in self.vtk_widgets:
            if vtk_widget.is_rotating:
                vtk_widget.stop_auto_rotation()
                self.autoRotateButton.setText("自动旋转")
            else:
                vtk_widget.start_auto_rotation()
                self.autoRotateButton.setText("停止旋转")

    def render_images(self):
        file1, _ = QFileDialog.getOpenFileName(self, "选择第一张图像", ".", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file1:
            try:
                pixmap1 = QPixmap(file1)
                if pixmap1.isNull():
                    raise ValueError("无法加载第一张图像")
                self.image_label1.setPixmap(pixmap1)
                self.image_label1.setText("")
                self.image_label1.adjustSize()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载第一张图像失败: {str(e)}")
                self.image_label1.setText("图像1加载失败")

        file2, _ = QFileDialog.getOpenFileName(self, "选择第二张图像", ".", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file2:
            try:
                pixmap2 = QPixmap(file2)
                if pixmap2.isNull():
                    raise ValueError("无法加载第二张图像")
                self.image_label2.setPixmap(pixmap2)
                self.image_label2.setText("")
                self.image_label2.adjustSize()
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载第二张图像失败: {str(e)}")
                self.image_label2.setText("图像2加载失败")


def main():
    app = QApplication(sys.argv)
    window = DemoWidget()
    window.resize(1000, 800)  # 调整窗口大小以适应上下布局
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()