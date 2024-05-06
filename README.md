# ME500-Final-Spring24
Takes STL as input, returns gcode for unidirectional 3d printing of a single layer for magnetically responsive structures.
Written for use with solenoid.
Print direction defined by self.print_angle, defaults to 0.

Within main.py adds functions:
calc_angle()
points_on_line()
find_intersection()
f1()
f2()
f3()
to Slicer class within mecode 

Limitations:
Cannot handle curved perimeters
Bug with printing perimeter when print direction is parallel to perimeter line

