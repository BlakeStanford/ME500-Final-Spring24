"""
Mecode
======

### GCode for all

Mecode is designed to simplify GCode generation. It is not a slicer, thus it
can not convert CAD models to 3D printer ready code. It simply provides a
convenient, human-readable layer just above GCode. If you often find
yourself manually writing your own GCode, then mecode is for you.

Basic Use
---------
To use, simply instantiate the `G` object and use its methods to trace your
desired tool path. ::

    from mecode import G
    g = G()
    g.move(10, 10)  # move 10mm in x and 10mm in y
    g.arc(x=10, y=5, radius=20, direction='CCW')  # counterclockwise arc with a radius of 20
    g.meander(5, 10, spacing=1)  # trace a rectangle meander with 1mm spacing between the passes
    g.abs_move(x=1, y=1)  # move the tool head to position (1, 1)
    g.home()  # move the tool head to the origin (0, 0)

By default `mecode` simply prints the generated GCode to stdout. If instead you
want to generate a file, you can pass a filename. ::

    g = G(outfile='path/to/file.gcode')

*NOTE:* When using the option direct_write=True or when writing to a file, 
`g.teardown()` must be called after all commands are executed. If you
are writing to a file, this can be accomplished automatically by using G as
a context manager like so:

```python
with G(outfile='file.gcode') as g:
    g.move(10)
```

When the `with` block is exited, `g.teardown()` will be automatically called.

The resulting toolpath can be visualized in 3D with
the `view()` method ::

    g = G()
    g.meander(10, 10, 1)
    g.view()

* *Author:* Jack Minardi
* *Email:* jack@minardi.org

Edits by: Javier Morales 
Email: javier96@bu.edu

Subclass by: Sophie Caplan
Email: scaplan@bu.edu


This software was developed by the Lewis Lab at Harvard University and Voxel8 Inc.

"""
import numpy as np
import math
import os
import string
from collections import defaultdict

from .printer import Printer

HERE = os.path.dirname(os.path.abspath(__file__))

# for python 2/3 compatibility
try:
    isinstance("", basestring)

    def is_str(s):
        return isinstance(s, basestring)

    def encode2To3(s):
        return s

    def decode2To3(s):
        return s

except NameError:

    def is_str(s):
        return isinstance(s, str)

    def encode2To3(s):
        return bytes(s, 'UTF-8')

    def decode2To3(s):
        return s.decode('UTF-8')


class G(object):

    def __init__(self, outfile=None, print_lines='auto', header=None, footer=None,
                 aerotech_include=False,
                 output_digits=6,
                 direct_write=False,
                 direct_write_mode='socket',
                 printer_host='localhost',
                 printer_port=8002,
                 baudrate=250000,
                 two_way_comm=True,
                 x_axis='X',
                 y_axis='Y',
                 z_axis='Z',
                 i_axis='I',
                 j_axis='J',
                 k_axis='K',
                 extrude=False,
                 filament_diameter=1.75,
                 layer_height=0.19,
                 extrusion_width=0.35,
                 extrusion_multiplier=1,
                 setup=True,
                 lineend='os'):
        """
        Parameters
        ----------
        outfile : path or None (default: None)
            If a path is specified, the compiled gcode will be writen to that
            file.
        print_lines : bool (default: 'auto')
            Whether or not to print the compiled GCode to stdout. If set to
            'auto' then lines will be printed if no outfile given.
        header : path or None (default: None)
            Optional path to a file containing lines to be written at the
            beginning of the output file
        footer : path or None (default: None)
            Optional path to a file containing lines to be written at the end
            of the output file.
        aerotech_include : bool (default: False)
            If true, add aerotech specific functions and var defs to outfile.
        output_digits : int (default: 6)
            How many digits to include after the decimal in the output gcode.
        direct_write : bool (default: False)
            If True a socket or serial port is opened to the printer and the
            GCode is sent directly over.
        direct_write_mode : str (either 'socket' or 'serial') (default: socket)
            Specify the channel your printer communicates over, only used if
            `direct_write` is True.
        printer_host : str (default: 'localhost')
            Hostname of the printer, only used if `direct_write` is True.
        printer_port : int (default: 8000)
            Port of the printer, only used if `direct_write` is True.
        baudrate: int (default: 250000)
            The baudrate to connect to the printer with.
        two_way_comm : bool (default: True)
            If True, mecode waits for a response after every line of GCode is
            sent over the socket. The response is returned by the `write`
            method. Only applies if `direct_write` is True.
        x_axis : str (default 'X')
            The name of the x axis (used in the exported gcode)
        y_axis : str (default 'Y')
            The name of the z axis (used in the exported gcode)
        z_axis : str (default 'Z')
            The name of the z axis (used in the exported gcode)
        i_axis : str (default 'I')
            The name of the i axis (used in the exported gcode)
        j_axis : str (default 'J')
            The name of the j axis (used in the exported gcode)
        k_axis : str (default 'K')
            The name of the k axis (used in the exported gcode)
        extrude : True or False (default: False)
            If True, a flow calculation will be done in the move command. The
            neccesary length of filament to be pushed through on a move command
            will be tagged on as a kwarg. ex. X5 Y5 E3
        filament_diameter: float (default 1.75)
            the diameter of FDM filament you are using
        layer_height : float
            Layer height for FDM printing. Only relavent when extrude = True.
        extrusion width: float
            total width of the capsule shaped cross section of a squashed filament.
        extrusion_multiplier: float (default = 1)
            The length of extrusion filament to be pushed through on a move
            command will be multiplied by this number before being applied.
        setup : Bool (default: True)
            Whether or not to automatically call the setup function.
        lineend : str (default: 'os')
            Line ending to use when writing to a file or printer. The special
            value 'os' can be passed to fall back on python's automatic
            lineending insertion.

        """
        self.outfile = outfile
        self.print_lines = print_lines
        self.header = header
        self.footer = footer
        self.aerotech_include = aerotech_include
        self.output_digits = output_digits
        self.direct_write = direct_write
        self.direct_write_mode = direct_write_mode
        self.printer_host = printer_host
        self.printer_port = printer_port
        self.baudrate = baudrate
        self.two_way_comm = two_way_comm
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.i_axis = i_axis
        self.j_axis = j_axis
        self.k_axis = k_axis

        self._current_position = defaultdict(float)
        self.is_relative = True

        self.extrude = extrude
        self.filament_diameter = filament_diameter
        self.layer_height = layer_height
        self.extrusion_width = extrusion_width
        self.extrusion_multiplier = extrusion_multiplier

        self.position_history = [(0, 0, 0)]
        self.speed = 0
        self.speed_history = []

        self._socket = None
        self._p = None

        # If the user passes in a line ending then we need to open the output
        # file in binary mode, otherwise python will try to be smart and
        # convert line endings in a platform dependent way.
        if lineend == 'os':
            mode = 'w+'
            self.lineend = '\n'
        else:
            mode = 'wb+'
            self.lineend = lineend

        if is_str(outfile):
            self.out_fd = open(outfile, mode)
        elif outfile is not None:  # if outfile not str assume it is an open file
            self.out_fd = outfile
        else:
            self.out_fd = None

        if setup:
            self.setup()

    @property
    def current_position(self):
        return self._current_position

    def __enter__(self):
        """
        Context manager entry
        Can use like:

        with mecode.G(  outfile=self.outfile,
                        print_lines=False,
                        aerotech_include=False) as g:
            <code block>
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit
        """
        self.teardown()

    # GCode Aliases  ########################################################

    def set_home(self, x=None, y=None, z=None, **kwargs):
        """ Set the current position to the given position without moving.

        Example
        -------
        >>> # set the current position to X=0, Y=0
        >>> g.set_home(0, 0)

        """
        args = self._format_args(x, y, z, **kwargs)
        space = ' ' if len(args) > 0 else ''
        self.write('G92' + space + args + ' ;set home');

        self._update_current_position(mode='absolute', x=x, y=y, z=z, **kwargs)

    def reset_home(self):
        """ Reset the position back to machine coordinates without moving.
        """
        # FIXME This does not work with internal current_position
        # FIXME You must call an abs_move after this to re-sync
        # current_position
        self.write('G92.1 ;reset position to machine coordinates without moving')

    def relative(self):
        """ Enter relative movement mode, in general this method should not be
        used, most methods handle it automatically.

        """
        if not self.is_relative:
            self.write('G91 ;relative')
            self.is_relative = True

    def absolute(self):
        """ Enter absolute movement mode, in general this method should not be
        used, most methods handle it automatically.

        """
        if self.is_relative:
            self.write('G90 ;absolute')
            self.is_relative = False


    def feed(self, rate):
        """ Set the feed rate (tool head speed) in (typically) mm/min

            If you use g.set_units_mm_s or g.set_units_mm_min it will 
            set the units of the g.feed function. 
        Parameters
        ----------
        rate : float
            The speed to move the tool head in (typically) mm/min.

        """
        self.write('G1 F{}'.format(rate))
        self.speed = rate

    def dwell(self, time):
        """ Pause code executions for the given amount of time.

        Parameters
        ----------
        time : float
            Time in seconds to pause code execution.

        """
        self.write('G4 P{}'.format(time))

    # Composed Functions  #####################################################

    def setup(self):
        """ Set the environment into a consistent state to start off. This
        method must be called before any other commands.

        """
        self._write_header()
        if self.is_relative:
            self.write('G91 ;relative')
        else:
            self.write('G90 ;absolute')

    def teardown(self, wait=True):
        """ Close the outfile file after writing the footer if opened. This
        method must be called once after all commands.

        Parameters
        ----------
        wait : Bool (default: True)
            Only used if direct_write_model == 'serial'. If True, this method
            waits to return until all buffered lines have been acknowledged.

        """
        if self.out_fd is not None:
            if self.aerotech_include is True:
               with open(os.path.join(HERE, 'footerv2.txt')) as fd:
                    self._write_out(lines=fd.readlines())
            if self.footer is not None:
                with open(self.footer) as fd:
                   self._write_out(lines=fd.readlines())
            if self.outfile is None:
                self.out_fd.close()
        if self._socket is not None:
            self._socket.close()
        if self._p is not None:
            self._p.disconnect(wait)

 
    def home(self):
        """ Move the tool head to the home position (X=0, Y=0).
        """
        self.abs_move(x = 0, y = 0) 

    def move(self, x=None, y=None, z=None, rapid=False, **kwargs):
        """ Move the tool head to the given position. This method operates in
        relative mode unless a manual call to `absolute` was given previously.
        If an absolute movement is desired, the `abs_move` method is
        recommended instead.

        Examples
        --------
        >>> # move the tool head 10 mm in x and 10 mm in y
        >>> g.move(x=10, y=10)
        >>> # the x, y, and z keywords may be omitted:
        >>> g.move(10, 10, 10)

        >>> # move the A axis up 20 mm
        >>> g.move(A=20)

        """
        if self.extrude is True and 'E' not in kwargs.keys():
            if self.is_relative is not True:
                x_move = self.current_position['x'] if x is None else x
                y_move = self.current_position['y'] if y is None else y
                x_distance = abs(x_move - self.current_position['x'])
                y_distance = abs(y_move - self.current_position['y'])
                current_extruder_position = self.current_position['E']
            else:
                x_distance = 0 if x is None else x
                y_distance = 0 if y is None else y
                current_extruder_position = 0
            line_length = math.sqrt(x_distance**2 + y_distance**2)
            area = self.layer_height*(self.extrusion_width-self.layer_height) + \
                3.14159*(self.layer_height/2)**2
            volume = line_length*area
            filament_length = ((4*volume)/(3.14149*self.filament_diameter**2))*self.extrusion_multiplier
            kwargs['E'] = filament_length + current_extruder_position

        self._update_current_position(x=x, y=y, z=z, **kwargs)
        args = self._format_args(x, y, z, **kwargs)
        cmd = 'G0 ' if rapid else 'G1 '
        self.write(cmd + args)

    def abs_move(self, x=None, y=None, z=None, rapid=False, **kwargs):
        """ Same as `move` method, but positions are interpreted as absolute.
        """
        if self.is_relative:
            self.absolute()
            self.move(x=x, y=y, z=z, rapid=rapid, **kwargs)
            self.relative()
        else:
            self.move(x=x, y=y, z=z, rapid=rapid, **kwargs)
            

    def rapid(self, x=None, y=None, z=None, **kwargs):
        """ Executes an uncoordinated move to the specified location.
        """
        self.move(x, y, z, rapid=True, **kwargs)

    def abs_rapid(self, x=None, y=None, z=None, **kwargs):
        """ Executes an uncoordinated abs move to the specified location.
        """
        self.abs_move(x, y, z, rapid=True, **kwargs)

    def retract(self, retraction):
        if self.extrude is False:
            self.move(E = -retraction)
        else:
            self.extrude = False
            self.move(E = -retraction)
            self.extrude = True

    def arc(self, x=None, y=None, z=None, direction='CW', radius='auto',
            helix_dim=None, helix_len=0, **kwargs):
        """ Arc to the given point with the given radius and in the given
        direction. If helix_dim and helix_len are specified then the tool head
        will also perform a linear movement through the given dimension while
        completing the arc.

        Parameters
        ----------
        points : floats
            Must specify two points as kwargs, e.g. x=5, y=5
        direction : str (either 'CW' or 'CCW') (default: 'CW')
            The direction to execute the arc in.
        radius : 'auto' or float (default: 'auto')
            The radius of the arc. A negative value will select the longer of
            the two possible arc segments. If auto is selected the radius will
            be set to half the linear distance to desired point.
        helix_dim : str or None (default: None)
            The linear dimension to complete the helix through
        helix_len : float
            The length to move in the linear helix dimension.

        Examples
        --------
        >>> # arc 10 mm up in y and 10 mm over in x with a radius of 20.
        >>> g.arc(x=10, y=10, radius=20)

        >>> # move 10 mm up on the A axis, arcing through y with a radius of 20
        >>> g.arc(A=10, y=0, radius=20)

        >>> # arc through x and y while moving linearly on axis A
        >>> g.arc(x=10, y=10, radius=50, helix_dim='A', helix_len=5)

        """
        dims = dict(kwargs)
        if x is not None:
            dims['x'] = x
        if y is not None:
            dims['y'] = y
        if z is not None:
            dims['z'] = z
        msg = 'Must specify two of x, y, or z.'
        if len(dims) != 2:
            raise RuntimeError(msg)
        dimensions = [k.lower() for k in dims.keys()]
        if 'x' in dimensions and 'y' in dimensions:
            plane_selector = 'G17 ;XY plane'  # XY plane
            axis = helix_dim
        elif 'x' in dimensions:
            plane_selector = 'G18 ;XZ plane'  # XZ plane
            dimensions.remove('x')
            axis = dimensions[0].upper()
        elif 'y' in dimensions:
            plane_selector = 'G19 ;YZ plane'  # YZ plane
            dimensions.remove('y')
            axis = dimensions[0].upper()
        else:
            raise RuntimeError(msg)
        if self.z_axis != 'Z':
            axis = self.z_axis

        if direction == 'CW':
            command = 'G2'
        elif direction == 'CCW':
            command = 'G3'

        values = [v for v in dims.values()]
        if self.is_relative:
            dist = math.sqrt(values[0] ** 2 + values[1] ** 2)
        else:
            k = [ky for ky in dims.keys()]
            cp = self._current_position
            dist = math.sqrt(
                (cp[k[0]] - values[0]) ** 2 + (cp[k[1]] - values[1]) ** 2
            )
        if radius == 'auto':
            radius = dist / 2.0
        elif abs(radius) < dist / 2.0:
            msg = 'Radius {} to small for distance {}'.format(radius, dist)
            #raise RuntimeError(msg)

        #extrude feature implementation
        # only designed for flow calculations in x-y plane
        if self.extrude is True:
            area = self.layer_height*(self.extrusion_width-self.layer_height) + 3.14159*(self.layer_height/2)**2
            if self.is_relative is not True:
                current_extruder_position = self.current_position['E']
            else:
                current_extruder_position = 0

            circle_circumference = 2*3.14159*abs(radius)

            arc_angle = ((2*math.asin(dist/(2*abs(radius))))/(2*3.14159))*360
            shortest_arc_length = (arc_angle/180)*3.14159*abs(radius)
            if radius > 0:
                arc_length = shortest_arc_length
            else:
                arc_length = circle_circumference - shortest_arc_length
            volume = arc_length*area
            filament_length = ((4*volume)/(3.14149*self.filament_diameter**2))*self.extrusion_multiplier
            dims['E'] = filament_length + current_extruder_position

        if axis is not None:
            self.write('G16 X Y {} ;coordinate axis assignment'.format(axis))  # coordinate axis assignment
        self.write(plane_selector)
        args = self._format_args(**dims)
        if helix_dim is None:
            self.write('{0} {1} R{2:.{digits}f}'.format(command, args, radius,
                                                        digits=self.output_digits))
        else:
            self.write('{0} {1} R{2:.{digits}f} G1 {3}{4}'.format(
                command, args, radius, helix_dim.upper(), helix_len, digits=self.output_digits))
            dims[helix_dim] = helix_len

        self._update_current_position(**dims)


    def arc_ijk(self, target, center, plane, direction='CW', helix_len=None):
        """ Arc to the given point with the given radius and in the given
        direction. If helix_dim and helix_len are specified then the tool head
        will also perform a linear movement along the axis orthogonal to the
        arc plane while completing the arc.

        Parameters
        ----------
        plane : str ('xy', 'yz', 'xz')
            Plane in which the arc is drawn
        target : 2-tuple of coordinates
            the end point of the arc, on the relevant plane
        center : 2-tuple of coordinates
            the distance to the center point of the arc from the
            starting position, on the relevant plane
        direction : str (either 'CW' or 'CCW') (default: 'CW')
            The direction to execute the arc in.
        helix_len : float
            The distance to move along the axis orthogonal to the arc plane
            during the arc.

        """

        if len(target) != 2:
            raise RuntimeError("'target' must be a 2-tuple of numbers (passed %s)" % target)
        if len(center) != 2:
            raise RuntimeError("'center' must be a 2-tuple of numbers (passed %s)" % center)

        if plane == 'xy':
            self.write('G17 ;XY plane')  # XY plane
            dims = {
                'x' : target[0],
                'y' : target[1],
                'i' : center[0],
                'j' : center[1],
            }
            if helix_len:
                dims['z'] = helix_len
        elif plane == 'yz':
            self.write('G19 ;YZ plane')  # YZ plane
            dims = {
                'y' : target[0],
                'z' : target[1],
                'j' : center[0],
                'k' : center[1],
            }
            if helix_len:
                dims['x'] = helix_len
        elif plane == 'xz':
            self.write('G18 ;XZ plane')  # XZ plane
            dims = {
                'x' : target[0],
                'z' : target[1],
                'i' : center[0],
                'k' : center[1],
            }
            if helix_len:
                dims['y'] = helix_len
        else:
            raise RuntimeError("Selected plane ('%s') is not one of ('xy', 'yz', 'xz')!" % plane)

        if direction == 'CW':
            command = 'G2'
        elif direction == 'CCW':
            command = 'G3'


        args = self._format_args(**dims)

        self.write('{} {}'.format(command, args))

        self._update_current_position(**dims)

    def abs_arc(self, direction='CW', radius='auto', **kwargs):
        """ Same as `arc` method, but positions are interpreted as absolute.
        """
        if self.is_relative:
            self.absolute()
            self.arc(direction=direction, radius=radius, **kwargs)
            self.relative()
        else:
            self.arc(direction=direction, radius=radius, **kwargs)

    def rect(self, x, y, direction='CW', start='LL'):
        """ Trace a rectangle with the given width and height.

        Parameters
        ----------
        x : float
            The width of the rectangle in the x dimension.
        y : float
            The height of the rectangle in the y dimension.
        direction : str (either 'CW' or 'CCW') (default: 'CW')
            Which direction to complete the rectangle in.
        start : str (either 'LL', 'UL', 'LR', 'UR') (default: 'LL')
            The start of the rectangle -  L/U = lower/upper, L/R = left/right
            This assumes an origin in the lower left.

        Examples
        --------
        >>> # trace a 10x10 clockwise square, starting in the lower left corner
        >>> g.rect(10, 10)

        >>> # 1x5 counterclockwise rect starting in the upper right corner
        >>> g.rect(1, 5, direction='CCW', start='UR')

        """
        if direction == 'CW':
            if start.upper() == 'LL':
                self.move(y=y)
                self.move(x=x)
                self.move(y=-y)
                self.move(x=-x)
            elif start.upper() == 'UL':
                self.move(x=x)
                self.move(y=-y)
                self.move(x=-x)
                self.move(y=y)
            elif start.upper() == 'UR':
                self.move(y=-y)
                self.move(x=-x)
                self.move(y=y)
                self.move(x=x)
            elif start.upper() == 'LR':
                self.move(x=-x)
                self.move(y=y)
                self.move(x=x)
                self.move(y=-y)
        elif direction == 'CCW':
            if start.upper() == 'LL':
                self.move(x=x)
                self.move(y=y)
                self.move(x=-x)
                self.move(y=-y)
            elif start.upper() == 'UL':
                self.move(y=-y)
                self.move(x=x)
                self.move(y=y)
                self.move(x=-x)
            elif start.upper() == 'UR':
                self.move(x=-x)
                self.move(y=-y)
                self.move(x=x)
                self.move(y=y)
            elif start.upper() == 'LR':
                self.move(y=y)
                self.move(x=-x)
                self.move(y=-y)
                self.move(x=x)

    def meander(self, x, y, spacing, start='LL', orientation='x', tail=False,
                minor_feed=None):
        """ Infill a rectangle with a square wave meandering pattern. If the
        relevant dimension is not a multiple of the spacing, the spacing will
        be tweaked to ensure the dimensions work out.

        Parameters
        ----------
        x : float
            The width of the rectangle in the x dimension.
        y : float
            The height of the rectangle in the y dimension.
        spacing : float
            The space between parallel meander lines.
        start : str (either 'LL', 'UL', 'LR', 'UR') (default: 'LL')
            The start of the meander -  L/U = lower/upper, L/R = left/right
            This assumes an origin in the lower left.
        orientation : str ('x' or 'y') (default: 'x')
        tail : Bool (default: False)
            Whether or not to terminate the meander in the minor axis
        minor_feed : float or None (default: None)
            Feed rate to use in the minor axis

        Examples
        --------
        >>> # meander through a 10x10 sqaure with a spacing of 1mm starting in
        >>> # the lower left.
        >>> g.meander(10, 10, 1)

        >>> # 3x5 meander with a spacing of 1 and with parallel lines through y
        >>> g.meander(3, 5, spacing=1, orientation='y')

        >>> # 10x5 meander with a spacing of 2 starting in the upper right.
        >>> g.meander(10, 5, 2, start='UR')

        """
        if start.upper() == 'UL':
            x, y = x, -y
        elif start.upper() == 'UR':
            x, y = -x, -y
        elif start.upper() == 'LR':
            x, y = -x, y

        # Major axis is the parallel lines, minor axis is the jog.
        if orientation == 'x':
            major, major_name = x, 'x'
            minor, minor_name = y, 'y'
        else:
            major, major_name = y, 'y'
            minor, minor_name = x, 'x'

        actual_spacing = self._meander_spacing(minor, spacing)
        if abs(actual_spacing) != spacing:
            msg = ';WARNING! meander spacing updated from {} to {}'
            self.write(msg.format(spacing, actual_spacing))
        spacing = actual_spacing
        sign = 1

        was_absolute = True
        if not self.is_relative:
            self.relative()
        else:
            was_absolute = False

        major_feed = self.speed
        if not minor_feed:
            minor_feed = self.speed
        for _ in range(int(self._meander_passes(minor, spacing))):
            self.move(**{major_name: (sign * major)})
            if minor_feed != major_feed:
                self.feed(minor_feed)
            self.move(**{minor_name: spacing})
            if minor_feed != major_feed:
                self.feed(major_feed)
            sign = -1 * sign
        if tail is False:
            self.move(**{major_name: (sign * major)})

        if was_absolute:
            self.absolute()

    def clip(self, axis='z', direction='+x', height=4):
        """ Move the given axis up to the given height while arcing in the
        given direction.

        Parameters
        ----------
        axis : str (default: 'z')
            The axis to move, e.g. 'z'
        direction : str (either +-x or +-y) (default: '+x')
            The direction to arc through
        height : float (default: 4)
            The height to end up at

        Examples
        --------
        >>> # move 'z' axis up 4mm while arcing through positive x
        >>> g.clip()

        >>> # move 'A' axis up 10mm while arcing through negative y
        >>> g.clip('A', height=10, direction='-y')

        """
        secondary_axis = direction[1]
        if height > 0:
            orientation = 'CW' if direction[0] == '-' else 'CCW'
        else:
            orientation = 'CCW' if direction[0] == '-' else 'CW'
        radius = abs(height / 2.0)
        kwargs = {
            secondary_axis: 0,
            axis: height,
            'direction': orientation,
            'radius': radius,
        }
        self.arc(**kwargs)

    def triangular_wave(self, x, y, cycles, start='UR', orientation='x'):
        """ Perform a triangular wave.

        Parameters
        ----------
        x : float
            The length to move in x in one half cycle
        y : float
            The length to move in y in one half cycle
        start : str (either 'LL', 'UL', 'LR', 'UR') (default: 'UR')
            The start of the zigzag direction.
            This assumes an origin in the lower left, and move toward upper
            right.
        orientation : str ('x' or 'y') (default: 'x')

        Examples
        --------
        >>> # triangular wave for one cycle going 10 in x and 10 in y per half
        >>> # cycle.
        >>> # the lower left.
        >>> g.zigzag(10, 10, 1)

        >>> # triangular wave 4 cycles, going 3 in x and 5 in y per half cycle,
        >>> # oscillating along the y axis.
        >>> g.zigzag(3, 5, 4, orientation='y')

        >>> # triangular wave 2 cycles, going 10 in x and 5 in y per half cycle,
        >>> # oscillating along the x axis making the first half cycle towards
        >>> # the lower left corner of the movement area.
        >>> g.zigzag(10, 5, 2, start='LL')

        """
        if start.upper() == 'UL':
            x, y = -x, y
        elif start.upper() == 'LL':
            x, y = -x, -y
        elif start.upper() == 'LR':
            x, y = x, -y

        # Major axis is the parallel lines, minor axis is the jog.
        if orientation == 'x':
            major, major_name = x, 'x'
            minor, minor_name = y, 'y'
        else:
            major, major_name = y, 'y'
            minor, minor_name = x, 'x'

        sign = 1

        was_absolute = True
        if not self.is_relative:
            self.relative()
        else:
            was_absolute = False

        for _ in range(int(cycles*2)):
            self.move(**{minor_name: (sign * minor), major_name: major})
            sign = -1 * sign

        if was_absolute:
            self.absolute()

    # AeroTech Specific Functions  ############################################

    def get_axis_pos(self, axis):
        """ Gets the current position of the specified `axis`.
        """
        cmd = 'AXISSTATUS({}, DATAITEM_PositionFeedback)'.format(axis.upper())
        pos = self.write(cmd)
        return float(pos)

    def set_cal_file(self, path):
        """ Dynamically applies the specified calibration file at runtime.

        Parameters
        ----------
        path : str
            The path specifying the aerotech calibration file.

        """
        self.write(r'LOADCALFILE "{}", 2D_CAL'.format(path))

    def toggle_pressure(self, com_port):
        self.write('Call togglePress P{}'.format(com_port))

    def set_pressure(self, com_port, value):
        self.write('Call setPress P{} Q{}'.format(com_port, value))

    def set_vac(self, com_port, value):
        self.write('Call setVac P{} Q{}'.format(com_port, value))

    def set_valve(self, num, value):
        self.write('$DO{}.0={}'.format(num, value))

    #### Javier add-ins ####


    def get_axis_pos_aero(self, axis):
        """ Gets the current position of the specified `axis` and defines this as a variable,
        Xnow, Ynow or Znow.
        """
        cmd = '${}now = AXISSTATUS({}, DATAITEM_PositionFeedback)'.format(axis.upper(),axis.upper())
        pos = self.write(cmd)
        

    def valve_open(self, com_port, valve_port):
        """
        Open solenoids valve, available valve_ports from 3 to 7 (five). Need to define the com 
        port of the arduino and which valve is on use.
        """
        self.write('Call pressure_on_D{} P{}'.format(valve_port,com_port))

    def valve_close(self, com_port , valve_port):
        """
        Close solenoids valve, available valve_ports from 3 to 7 (five). Need to define the com 
        port of the arduino and which valve is on use.
        """
        self.write('Call pressure_off_D{} P{}'.format(valve_port,com_port))

    def rel_move(self, x=None, y=None, z=None, rapid=False, **kwargs):
        """ Same as `move` method, but positions are interpreted as relative.
        """
        self.relative()
        self.move(x=x, y=y, z=z, rapid=rapid, **kwargs)
        self.absolute()
        
    def circle(self,radius,resolution,xstep,ystep):
        """
        This fucntion generated a circle, inputs are radius, the resoltion for the curve, xstep, ystep.
        xstep and ystep are use if you need to make arrays or circles, if not set them to zero. 
        Thisfunction discretize the curve by the a certain number or the resolution.
        Recommended resolution 15 - 20 ** This will be vary with the dimmentions of the curve

        """

        theta = np.linspace(0,  2* np.pi, resolution)
        x = (radius) * np.cos(theta)
        y = (radius) * np.sin(theta)
        if abs(y[0] - y[-1]) < 0.0001:
            y[-1] = y[0]

        for xpt, ypt in zip(x, y):
            self.move(xpt+xstep, ypt+ystep)
        
    # ramp up - acceleration and deceleration modes 
    def ramp_zero(self, time = 0):

        """ 
        This function allows to define an specific acceleration and deceleration rate for the begin and ending of a path. 
        With this function it will be set to zero by default. Still you have the ability to change it. 

        The usefulness of this is that you can control the intial and final steps of the print avoiding imperfactions on the print path. 

        Default ramp time in motion system is 0.5 
        """ 
        self.write('G67 ; Activating time-based ramp mode ')
        self.write('RAMP TIME {} ; Setting ramp time'.format(time))


    # Units mm/s
    def set_units_mm_s(self):
        """ 
        This funtion change the speed units to mm/s, the max velocity is 44.9 mm/s.
        
        """
        self.write(';Setting units to mm/s')
        self.write('G76')



    # Units mm/min
    def set_units_mm_min(self):
        """ 
        This funtion change the speed units to mm/min, the max velocity is 2,699.4 mm/min.
        The printer by default is on mm/min. 
        """
        self.write(';Setting units to mm/min')
        self.write('G75')

    # FOR loop
    def for_Start(self, val_name='$FORLOOP', starting_value=0, ending_value=1, step=1):
        """
        This funtion is a translation of a for loop in the aerobasic language. It is useful if the print 
        path is to long (thosands of lines). Using this function will reduce the size of the output file. 

        For propertly functionality, when you want to close the loop you need to use g.for_End()

        val_name is the name of the task on aerobasic, no need to change it. 
        starting_value is the first iteration value.
        ending_point is the last iteration value. 
        step is how it count the steps between iterations. 

        The loop by default start on index 0 (as python), therefore if you want 6 loops (starting on 0) 
        you need to use an ending point of 5.

        ** the vizualizer or matpotlib cannot show this output, if you want to use those tools you need 
        to use the for loop of python and then translate to this function ** 
        """
        #self.write(';Define Variable')
        #self.write('DVAR {}'.format(val_name))
        self.write('; Starting for loop')
        self.write('FOR {} = {} TO {} STEP {}'.format(val_name, starting_value, ending_value, step))
    
    def for_End(self, val_name='$FORLOOP'):
        """
        Close the for loop. 
        """
        self.write('; Ending for loop')
        self.write('NEXT {}'.format(val_name))

    # WHILE loop

    def while_Start(self, val_name='$OUTER', condition = '<', condition_value=1):
        """
        This funtion is a translation of a while loop in the aerobasic language. It is useful if the print 
        path is to long (thosands of lines). Using this function will reduce the size of the output file. 

        For propertly functionality, when you want to close the loop you need to use g.while_End()
        
        val_name is the task name in aerobasic, no need to change it. 
        condition can it be >, <, => , =< 
        condition_value is the value used to apply the condition. 

        
        ** the vizualizer or matpotlib cannot show this output, if you want to use those tools you need 
        to use the for loop of python and then translate to this function ** 
        """
        self.write(';Define Variable')
        #self.write('DVAR {}'.format(val_name))
        self.write('{} = 0'.format(val_name))
        self.write('; Starting while loop')
        self.write('WHILE {} {} {} DO'.format(val_name, condition, condition_value))
    
    def reindent(s, numSpaces):
        s = string.split(s, '\n')
        s = [(numSpaces * ' ') + string.lstrip(line) for line in s]
        s = string.join(s, '\n')
        return s

    def while_End(self, val_name='$OUTER'):
        self.write('{} = {} + 1'.format(val_name,val_name))
        self.write('; Ending while loop')
        self.write('ENDWHILE')

  ###Evan PSO Functions###  

        #PSO control functions
    # Enable the PSO pulse generator.
    #def pressure_on_pso(self, axis):
    #   self.write('PSOCONTROL {} ON'.format(axis))

    # Disable the PSO pulse generator.
    def pressure_off_pso(self, axis):
        self.write('PSOCONTROL {} OFF'.format(axis))

    # Arm the PSO to start the distance/window tracking.
    def arm_pso(self, axis):
        self.write('PSOCONTROL {} ARM'.format(axis))
    
    # produce repeating pulses until psocontrol off argument is given
    def pressure_on_pso(self, axis):
        self.write('PSOCONTROL {} FIRE CONTINUOUS'.format(axis))

    # resets the pso logic, clearing previous data
    def reset_pso(self, axis):
        self.write('PSOCONTROL {} RESET'.format(axis))


    ###PSO SETUP FUNCTIONS###
    # enable the PSO hardware to respond to high-speed inputs.
    def pso_fast(self, axis):
        self.write('PSOHALT {}'.format(axis))
        self.write('PSOHALT {} INPUT 1 1'.format(axis))
    
    # specify conditions under which the PSO distance tracking counters will be reset automatically
    def pso_track_reset(self, axis, bitmask):
        self.write('PSOTRACK {} RESET {}'.format(axis, bitmask))

    # allows resolutions to be normalized so that vector distances will be correct when tracking axes with different resolutions
    def pso_track_scale(self, axis, scale):
        self.write('PSOTRACK {} SCALE {}'.format(axis, scale))

    # specify the feedback sources to be used for distance-based tracking(0 for x, 1 and 2 for y, 3 for z(goes up to 8))
    def pso_track_input(self, axis, encoder1):
        self.write('PSOTRACK {} INPUT {}'.format(axis, encoder1))
    
    # restrict firing events to one encoder direction only. You can only use this command with single axis tracking; optional
    def pso_track_direction(self, axis, bitmask):
        self.write('PSOTRACK {} DIRECTION {}'.format(axis, bitmask))#bitmask=1:no positive direction, bitmask=2:no negative direction

    def toggle_pso(self, axis):
        self.write('PSOOUTPUT {} TOGGLE'.format(axis))

    # resets the pso conditions and sets the pso output to read from the pulse generator hardware with a duty cycle of 100%
    def pso_pulse(self, axis):
        self.write('PSOPULSE {} TIME 1000, 1000'.format(axis))
        self.write('PSOOUTPUT {} PULSE'.format(axis))
    
    # resets the pso conditions and sets the pso output to read from the window hardware
    def pso_window(self, axis):
        self.write('PSOOUTPUT {} WINDOW'.format(axis))
    #
    def pso_output_control(self, axis, pin, mode):
        self.write('PSOOUTPUT {} CONTROL {} {}'.format(axis, pin, mode))
    
    ###PSO WINDOW FIRING###
    # creates the number of windows that the PSO will fire in
    def set_windows(self, num_windows):
        self.write('#define NUM_WINDOWS {}'.format(num_windows))
        self.write('DVAR $ArrayWindows[NUM_WINDOWS * 2]')

    # Fill the local array with the window range values.  The array contains the lower and upper bounds for the window ranges
    def array_value(self, arraynumber, axis, value):
        self.write('$ArrayWindows[{}] = ABS(UNITSTOCOUNTS({}, {}))'.format(arraynumber, axis, value))

    #Choose the window(1 or 2) and the encoder to use(0 for x, 1 or 2 for y, 3 for z(goes up to 8))// 6 for X-axis and 7 for Y-axis
    def pick_window(self, axis, windownumber, source):
        self.write('PSOWINDOW {} {} INPUT {}'.format(axis, windownumber, source))

    # Write the window ranges from the local variables into the drive array.
    def array_window_range(self, axis):
        self.write('ARRAY {} WRITE $ArrayWindows[0] 0 (NUM_WINDOWS * 2)'.format(axis))
    
    # Disable the PSO window hardware
    def turn_off_window(self, axis, window_number):
        self.write('PSOWINDOW {} {} OFF'.format(axis, window_number))

    # Configure the window hardware to get the upper and lower bounds from the drive array
    def set_pso_range(self, axis):
        self.write('PSOWINDOW {} 1 RANGE ARRAY 0 (NUM_WINDOWS * 2) EDGE 1'.format(axis))

    # load the window counter with a specific value; resets the absolute position reference for the PSO window
    def pso_load_window_value(self, axis, window, value):
        self.write('PSOWINDOW {} {} LOAD {}'.format(axis, window, value))

    #
    def fixed_window_range(self, axis, window, upperbound, lowerbound):
        self.write('PSOWINDOW {} {} RANGE {}, {}'.format(axis, window, upperbound, lowerbound))


    ###PSO FIXED DISTANCE FUNCTIONS###
    # a firing will occur each time the axis moves the distance specified by the firedistance; firedistance < 8,388,607
    def pso_distance_fixed(self, axis, firedistance, cycles):
        self.write('PSODISTANCE {} FIXED ABS(UNITSTOCOUNTS({}, {})) {}'.format(axis, axis, firedistance, cycles))

    # specify the distances between firing events based on data contained within the PSO array
    def pso_distance_array(self, axis, startindex, cycles):
        self.write('PSODISTANCE {} ARRAY ABS(UNITSTOCOUNTS({}, {})) {}'.format(axis, axis, startindex, cycles))



    # Alicat Specific Functions  ############################################

    def set_pressure_alicat(self, com_port, value):
        self.write('Call setPress_alicat P{} Q{}'.format(com_port, value))
    # Public Interface  #######################################################

    def close_valve_alicat(self, com_port):
        self.write('Call close_valve_alicat P{}'.format(com_port))

    def open_valve_alicat(self, com_port):
        self.write('Call open_valve_alicat P{}'.format(com_port))


    def view(self, backend='matplotlib'):
        """ View the generated Gcode.

        Parameters
        ----------
        backend : str (default: 'matplotlib')
            The plotting backend to use, one of 'matplotlib' or 'mayavi'.

        """
        import numpy as np
        history = np.array(self.position_history)

        if backend == 'matplotlib':
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y, Z = history[:, 0], history[:, 1], history[:, 2]
            ax.plot(X, Y, Z)

            # Hack to keep 3D plot's aspect ratio square. See SO answer:
            # http://stackoverflow.com/questions/13685386
            max_range = np.array([X.max()-X.min(),
                                  Y.max()-Y.min(),
                                  Z.max()-Z.min()]).max() / 2.0

            mean_x = X.mean()
            mean_y = Y.mean()
            mean_z = Z.mean()
            ax.set_xlim(mean_x - max_range, mean_x + max_range)
            ax.set_ylim(mean_y - max_range, mean_y + max_range)
            ax.set_zlim(mean_z - max_range, mean_z + max_range)

            plt.show()
        elif backend == 'mayavi':
            from mayavi import mlab
            mlab.plot3d(history[:, 0], history[:, 1], history[:, 2])
        else:
            raise Exception("Invalid plotting backend! Choose one of mayavi or matplotlib.")

    def write(self, statement_in, resp_needed=False):
        if self.print_lines or (self.print_lines == 'auto' and self.outfile is None):
            print(statement_in)
        self._write_out(statement_in)
        statement = encode2To3(statement_in + self.lineend)
        if self.direct_write is True:
            if self.direct_write_mode == 'socket':
                if self._socket is None:
                    import socket
                    self._socket = socket.socket(socket.AF_INET,
                                                socket.SOCK_STREAM)
                    self._socket.connect((self.printer_host, self.printer_port))
                self._socket.send(statement)
                if self.two_way_comm is True:
                    response = self._socket.recv(8192)
                    response = decode2To3(response)
                    if response[0] != '%':
                        raise RuntimeError(response)
                    return response[1:-1]
            elif self.direct_write_mode == 'serial':
                if self._p is None:
                    self._p = Printer(self.printer_port, self.baudrate)
                    self._p.connect()
                    self._p.start()
                if resp_needed:
                    return self._p.get_response(statement_in)
                else:
                    self._p.sendline(statement_in)

    def rename_axis(self, x=None, y=None, z=None):
        """ Replaces the x, y, or z axis with the given name.

        Examples
        --------
        >>> g.rename_axis(z='A')

        """
        if x is not None:
            self.x_axis = x
        elif y is not None:
            self.y_axis = y
        elif z is not None:
            self.z_axis = z
        else:
            msg = 'Must specify new name for x, y, or z only'
            raise RuntimeError(msg)

    # Private Interface  ######################################################

    def _write_out(self, line=None, lines=None):
        """ Writes given `line` or `lines` to the output file.
        """
        # Only write if user requested an output file.
        if self.out_fd is None:
            return

        if lines is not None:
            for line in lines:
                self._write_out(line)
                
        line = line.rstrip() + self.lineend  # add lineend character
        
        if hasattr(self.out_fd, 'mode') and 'b' in self.out_fd.mode:  # encode the string to binary if needed
            line = encode2To3(line)
        self.out_fd.write(line)


    def _meander_passes(self, minor, spacing):
        if minor > 0:
            passes = math.ceil(minor / spacing)
        else:
            passes = abs(math.floor(minor / spacing))
        return passes

    def _meander_spacing(self, minor, spacing):
        return minor / self._meander_passes(minor, spacing)

    def _write_header(self):
        if self.aerotech_include is True:
            with open(os.path.join(HERE, 'header.txt')) as fd:
                self._write_out(lines=fd.readlines())
        if self.header is not None:
            with open(self.header) as fd:
                self._write_out(lines=fd.readlines())

    def _format_args(self, x=None, y=None, z=None, i=None, j=None, k=None, **kwargs):
        d = self.output_digits
        args = []
        if x is not None:
            args.append('{0}{1:.{digits}f}'.format(self.x_axis, x, digits=d))
        if y is not None:
            args.append('{0}{1:.{digits}f}'.format(self.y_axis, y, digits=d))
        if z is not None:
            args.append('{0}{1:.{digits}f}'.format(self.z_axis, z, digits=d))
        if i is not None:
            args.append('{0}{1:.{digits}f}'.format(self.i_axis, i, digits=d))
        if j is not None:
            args.append('{0}{1:.{digits}f}'.format(self.j_axis, j, digits=d))
        if k is not None:
            args.append('{0}{1:.{digits}f}'.format(self.k_axis, k, digits=d))
        args += ['{0}{1:.{digits}f}'.format(k, kwargs[k], digits=d) for k in sorted(kwargs)]
        args = ' '.join(args)
        return args

    def _update_current_position(self, mode='auto', x=None, y=None, z=None,
                                 **kwargs):
        if mode == 'auto':
            mode = 'relative' if self.is_relative else 'absolute'

        if self.x_axis is not 'X' and x is not None:
            kwargs[self.x_axis] = x
        if self.y_axis is not 'Y' and y is not None:
            kwargs[self.y_axis] = y
        if self.z_axis is not 'Z' and z is not None:
            kwargs[self.z_axis] = z

        if mode == 'relative':
            if x is not None:
                self._current_position['x'] += x
            if y is not None:
                self._current_position['y'] += y
            if z is not None:
                self._current_position['z'] += z
            for dimention, delta in kwargs.items():
                self._current_position[dimention] += delta
        else:
            if x is not None:
                self._current_position['x'] = x
            if y is not None:
                self._current_position['y'] = y
            if z is not None:
                self._current_position['z'] = z
            for dimention, delta in kwargs.items():
                self._current_position[dimention] = delta

        x = self._current_position['x']
        y = self._current_position['y']
        z = self._current_position['z']

        self.position_history.append((x, y, z))

        len_history = len(self.position_history)
        if (len(self.speed_history) == 0
            or self.speed_history[-1][1] != self.speed):
            self.speed_history.append((len_history - 1, self.speed))


################################################################################################################################################
##### Sophie add-ins #####

from stl import mesh # mesh allows us to create a mesh from the STL file
from mpl_toolkits import mplot3d # for plotting
from matplotlib import pyplot # for plotting

class Slicer(G):
    """
        Slicer class inherits from G class.

        Input options:
            STL file (from CAD software... sketch on TOP plane in Onshape; FRONT plane in SOLIDWORKS)
            PGM file (written directly in g-code or me-code)

        Alignment options (i.e. algorithm number): 
            1 : very simple shapes ; uses distance formula
                works: circle, cube, cube_a, diamond, triangle
                doesn't work: any dogbone, star (more complicated)
            2 : follows coordinates by order of increasing angle to calculate outline
                works: star, star2, triangle, cube, cube_a
                doesn't work: any dogbone
            4 : DOG BONE ; cuts object in half and mirrors over y-axis
                works: dogbone, dogbone_a, dogbone_a2, dogbone_w2, dogbone_d, dogbone_e  ** must be symmetric about y-axis
                doesn't work: cube(too few coordinate points to divide into quarters), circle (use calc_base1)

        When calling a Slicer object, make sure to:
            object_name.set_pressure(24,10)  # sets the com_port and the pressure values
            object_name.set_units_mm_s()     # sets units
            object_name.feed(30)             # sets speed
            object_name.slicer()             # calls slicer function
            object_name.view()               # view in matplotlib
    """
    def __init__(self, outfile=None, print_lines='auto', header=None, footer=None,
                 aerotech_include=False,
                 output_digits=6,
                 direct_write=False,
                 direct_write_mode='socket',
                 printer_host='localhost',
                 printer_port=8002,
                 baudrate=250000,
                 two_way_comm=True,
                 x_axis='X',
                 y_axis='Y',
                 z_axis='Z',
                 i_axis='I',
                 j_axis='J',
                 k_axis='K',
                 extrude=False,
                 filament_diameter=1.75,
                 layer_height=0.19,
                 extrusion_width=0.35,
                 extrusion_multiplier=1,
                 setup=True,
                 lineend='os', # down to here is what was from G class itself
                 stl_file=None, # if using a cad STL file
                 pgm_file=None, # if using g-code directly
                 algorithm_num=None, # whether user wants 1,2,or 4; if None -> g-code
                 alignment='t', # tangential(vertical) or longitudinal (horizontal)
                 num_layers=None, # if None -> use stl file data
                 nozzle_size=.2, # common nozzle sizes are 0.2 and 0.41
                 spacing=0.95, # sp_XYZ overlapping factor
                 pressure_box="Nordson", # specify which pressure box is being used
                 com_port=24,
                 valve_port=4, # only used for Solenoid pressure box details
                 pressure=10,
                 print_speed=30,
                 is_circle=False,
                 tiz=0,
                 tfz=0,
                 print_angle=0):

        super(Slicer,self).__init__(outfile=None, print_lines='auto', header=None, footer=None,
                        aerotech_include=False,
                        output_digits=6,
                        direct_write=False,
                        direct_write_mode='socket',
                        printer_host='localhost',
                        printer_port=8002,
                        baudrate=250000,
                        two_way_comm=True,
                        x_axis='X',
                        y_axis='Y',
                        z_axis='Z',
                        i_axis='I',
                        j_axis='J',
                        k_axis='K',
                        extrude=False,
                        filament_diameter=1.75,
                        layer_height=0.19,
                        extrusion_width=0.35,
                        extrusion_multiplier=1,
                        setup=True,
                        lineend='os')
        self.stl_file=stl_file
        self.pgm_file=pgm_file
        self.algorithm_num=algorithm_num
        self.alignment=alignment
        self.num_layers=num_layers
        self.nozzle_size=nozzle_size
        self.spacing=spacing
        self.pressure_box=pressure_box
        self.com_port=com_port
        self.valve_port=valve_port
        self.pressure=pressure
        self.print_speed=print_speed
        self.is_circle=is_circle
        self.outfile = outfile
        self.print_lines = print_lines
        self.header = header
        self.footer = footer
        self.aerotech_include = aerotech_include
        self.output_digits = output_digits
        self.direct_write = direct_write
        self.direct_write_mode = direct_write_mode
        self.printer_host = printer_host
        self.printer_port = printer_port
        self.baudrate = baudrate
        self.two_way_comm = two_way_comm
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.i_axis = i_axis
        self.j_axis = j_axis
        self.k_axis = k_axis

        self._current_position = defaultdict(float)
        self.is_relative = True

        self.extrude = extrude
        self.filament_diameter = filament_diameter
        self.layer_height = layer_height
        self.extrusion_width = extrusion_width
        self.extrusion_multiplier = extrusion_multiplier

        self.position_history = [(0, 0, 0)]
        self.speed = 0
        self.speed_history = []

        self._socket = None
        self._p = None

        self.tiz=tiz
        self.tfz=tfz
        self.print_angle=print_angle

        if lineend == 'os':
            mode = 'w+'
            self.lineend = '\n'
        else:
            mode = 'wb+'
            self.lineend = lineend

        if is_str(outfile):
            self.out_fd = open(outfile, mode)
        elif outfile is not None:  # if outfile not str assume it is an open file
            self.out_fd = outfile
        else:
            self.out_fd = None

        if setup:
            self.setup()
    
    # Nordson Pressure Box - updated for switch
    # Beibit: valve+port = 0
    def toggle_pressure(self, com_port,valve_port=0):
        self.write('Call togglePress P{}'.format(com_port))

    # Alicat Pressure Box - updated for switch
    def close_valve_alicat(self, com_port,valve_port):
        self.write('Call close_valve_alicat P{}'.format(com_port))

    def open_valve_alicat(self, com_port,valve_port):
        self.write('Call open_valve_alicat P{}'.format(com_port))

    # Switch Dictionary - Pressure Box
    def toggle_valve(self, pressure_box, action, com_port,valve_port):
        """ Depending on which pressure box is being used, the correct fuctions are called to open/close the valve.
            Dictionary version of a switch since Python 2.7 does not support switches. """
        cases = {
            ("Alicat","open"): self.open_valve_alicat,
            ("Alicat","close"): self.close_valve_alicat,
            ("Nordson","toggle"): self.toggle_pressure,
            ("Solenoid", "open"): self.valve_open,
            ("Solenoid", "close"): self.valve_close,
            ("Solenoid2", "open"): self.pressure_on_pso,
            ("Solenoid2", "close"): self.pressure_off_pso
        }
        print('com_port: ', com_port,  'valve_port: ', valve_port)
        if pressure_box == 'Solenoid2':
            return cases[pressure_box,action]('X')
        else:
            return cases[pressure_box,action](com_port,valve_port)

    def action_switch(self,action_counter):
        if action_counter == -1:
            action = "close"
        elif action_counter == 1:
            action = "open"
        else:
            action = "toggle"
        return action
        

    def calc_dist(self,x1,y1,x2,y2):
        """calculates distance using distance formula. input coordinate values"""
        a = (x2 - x1) ** 2
        b = (y2 - y1) ** 2
        c = a + b
        d = math.sqrt(c)
        return d

    def find_closest(self,coord,others):
        """taking as input a set of (x,y) coordinates and an array of other coordinates, calc_dist to find 
        the closest coordinate in the array to the given coord.
            - coord is a numpy array [x y]
            - others is a numpy array of many smaller arrays of coords (like a list of lists)
        returns index in others of the set of closest coordinates"""
        x1 = coord[0]
        y1 = coord[1]
        smallest_dist = self.calc_dist(x1,y1,others[0][0],others[0][1])
        ind = 0
        ind_closest = 0
        for s in others:
            x2 = s[0]
            y2 = s[1]
            if self.calc_dist(x1,y1,x2,y2) < smallest_dist:
                smallest_dist = self.calc_dist(x1,y1,x2,y2)
                ind_closest = ind
            ind += 1
        return ind_closest
    
    def calc_theta(self,x,y):
        """takes as input x and y cartesian coord values and calculates and returns polar angle theta"""
        self.theta = math.degrees(math.atan2(y,x))
        if self.theta < 0:
            self.theta = 180 + (180+self.theta)
        return self.theta

    def add_equ_y_equals(self,coord_arr1, coord_arr2):
        """Calculates equation of a line and adds m and b (slope and y-intercept) to a dictionary called line_dict.
                Keys: x-coordinate range btwn points.
                Values: list of arrays of [m,b]
            Input of 2 arrays each with an x and y coordinate. 
            Returns line_dict. """
        x1 = coord_arr1[0]
        y1 = coord_arr1[1]
        x2 = coord_arr2[0]
        y2 = coord_arr2[1]

        if (x2 - x1) != 0:
            m = (y2-y1) / (x2-x1)
            m = round(m,9)
            b = y1-(m*x1)
            b = round(b,9)

            line = np.array([m,b])
        
            if x1 < x2:
                rangex = (x1,x2)
            else:
                rangex = (x2,x1)
            
            if rangex not in self.line_dict:
                self.line_dict[rangex] = list()

            counter = 0
            for l in self.line_dict[rangex]:
                if l[0] == line[0] and l[1] == line[1]:
                    counter += 1
            if counter == 0:
                self.line_dict[rangex].append(line)

        return self.line_dict
    
    def add_equ_x_equals(self,coord_arr1, coord_arr2):
        """ Calculates equation of a line and adds m and b (slope and y-intercept) to a dictionary called line_dict.
           Keys: y-coordinate range btwn points.
           Values: list of arrays of [m,b] OR x value of vertical line
        Input of 2 arrays each with an x and y coordinate. 
        Returns line_dict """
        x1 = coord_arr1[0]
        y1 = coord_arr1[1]
        x2 = coord_arr2[0]
        y2 = coord_arr2[1]

        if x1 == x2: # this means we have a vertical line and the slope (m) is undefined
            x1 = round(x1,9)
            line = np.array([x1]) # set the equation of the line equal to just the x-value
        else:
            m = (y2-y1) / (x2-x1)
            b = y1-(m*x1)
            m = round(m,9)
            b = round(b,9)
            line = np.array([m,b]) # line is a numpy array
        
        y1 = round(y1, 14) # round range values so that we don't get weird exponential values close to 0
        y2 = round(y2, 14)

        if y1 < y2:
            rangey = (y1,y2)
        else:
            rangey = (y2,y1)
        
        if rangey not in self.line_dict:
            self.line_dict[rangey] = list() # creates a new rangey KEY for the dictionary and the value type is list

        counter = 0
        for l in self.line_dict[rangey]:
            if len(l) == 1:
                if l[0] == line[0]:
                    counter += 1
            else:
                if l[0] == line[0] and l[1] == line[1]:
                    counter += 1
        if counter == 0:
            self.line_dict[rangey].append(line)

        return self.line_dict  
    
    def g_arc_t(self,center_pt,start_pt,end_pt,direction,radius,resolution):
        """
        Input the center coordinate of the arc, the starting point and ending point coords
        and what radius you want; direction 1 or -1 for CW or CCW; the resolution (higher res = smoother arc).
        Determines coordinate points along arc and then enters them into LINE_DICT as equations of lines.
            EX.
            center_pt = [15,30]
            start_pt = [10,30]
            end_pt = [20,30]
            direction = 1
            radius = 5
            resolution = 40 """
        theta = np.linspace(0,  2* np.pi, resolution)
        x = (radius) * np.cos(theta)
        y = (radius) * np.sin(theta)
        if abs(y[0] - y[-1]) < 0.0001:
            y[-1] = y[0]

        x_c = np.full(shape=len(theta),fill_value=center_pt[0],dtype=float) # just a numpy array of the center x-coord repeated
        y_c = np.full(shape=len(theta), fill_value=center_pt[1],dtype=float)

        x_coords = x_c+x
        y_coords = y_c+y

        z = [list(x) for x in zip(x_coords, y_coords)]
        # finding starting point in z list
        dist_x = abs(z[0][0] - start_pt[0])
        dist_y = abs(z[0][1] - start_pt[1])
        start_here = 0
        for i in range(len(z)-1):
            if abs(z[i][0] - start_pt[0]) <= dist_x:
                if abs(z[0][1] - start_pt[1]) <= dist_y:
                    dist_x = abs(z[i][0] - start_pt[0])
                    dist_y = abs(z[0][1] - start_pt[1])
                    start_here = i
        # finding ending point in z list
        dist_x = abs(z[0][0] - end_pt[0])
        dist_y = abs(z[0][1] - end_pt[1])
        end_here = 0
        for i in range(len(z)):
            if abs(z[i][0] - end_pt[0]) <= dist_x:
                if abs(z[0][1] - end_pt[1]) <= dist_y:
                    dist_x = abs(z[i][0] - end_pt[0])
                    dist_y = abs(z[0][1] - end_pt[1])
                    end_here = i

        z[start_here] = start_pt
        z[end_here] = end_pt

        ### if we're going clockwise starting point will be later in z than ending point
        if direction == 1: # CW
            z.reverse()
            z = z[start_here:end_here+2]
        else: # CCW
            z = z[start_here:end_here+1]
    
        for i in range(len(z)-2):
            ca1 = [z[i][0],z[i][1]]
            ca2 = [z[i+1][0],z[i+1][1]]
            if i == 0:
                starting=[start_pt[0],start_pt[1]]
                self.add_equ_y_equals(starting,ca1)
            self.add_equ_y_equals(ca1,ca2)
            if i == len(z)-3:
                ending = [end_pt[0],end_pt[1]]
                self.add_equ_y_equals(ca2,ending)
    
    def g_arc_l(self,center_pt,start_pt,end_pt,direction,radius,resolution):
        """
        Input the center coordinate of the arc, the starting point and ending point coords
        and what radius you want; direction 1 or -1 for CW or CCW; the resolution (higher res = smoother arc).
        Determines coordinate points along arc and then enters them into LINE_DICT as equations of lines.
            EX.
            center_pt = [15,30]
            start_pt = [10,30]
            end_pt = [20,30]
            direction = 1
            radius = 5
            resolution = 40 """
        theta = np.linspace(0,  2* np.pi, resolution)
        x = (radius) * np.cos(theta)
        y = (radius) * np.sin(theta)
        if abs(y[0] - y[-1]) < 0.0001:
            y[-1] = y[0]

        x_c = np.full(shape=len(theta),fill_value=center_pt[0],dtype=float) # just a numpy array of the center x-coord repeated
        y_c = np.full(shape=len(theta), fill_value=center_pt[1],dtype=float)

        x_coords = x_c+x
        y_coords = y_c+y

        z = [list(x) for x in zip(x_coords, y_coords)]
        # finding starting point in z list
        dist_x = abs(z[0][0] - start_pt[0])
        dist_y = abs(z[0][1] - start_pt[1])
        start_here = 0
        for i in range(len(z)-1):
            if abs(z[i][0] - start_pt[0]) <= dist_x:
                if abs(z[0][1] - start_pt[1]) <= dist_y:
                    dist_x = abs(z[i][0] - start_pt[0])
                    dist_y = abs(z[0][1] - start_pt[1])
                    start_here = i
        # finding ending point in z list
        dist_x = abs(z[0][0] - end_pt[0])
        dist_y = abs(z[0][1] - end_pt[1])
        end_here = 0
        for i in range(len(z)):
            if abs(z[i][0] - end_pt[0]) <= dist_x:
                if abs(z[0][1] - end_pt[1]) <= dist_y:
                    dist_x = abs(z[i][0] - end_pt[0])
                    dist_y = abs(z[0][1] - end_pt[1])
                    end_here = i

        z[start_here] = start_pt
        z[end_here] = end_pt

        ### if we're going clockwise starting point will be later in z than ending point
        if direction == 1: # CW
            z.reverse()
            z = z[start_here:end_here+2]
        else: # CCW
            z = z[start_here:end_here+1]
    
        for i in range(len(z)-2):
            ca1 = [z[i][0],z[i][1]]
            ca2 = [z[i+1][0],z[i+1][1]]
            if i == 0:
                starting=[start_pt[0],start_pt[1]]
                self.add_equ_x_equals(starting,ca1)
            self.add_equ_x_equals(ca1,ca2)
            if i == len(z)-3:
                ending = [end_pt[0],end_pt[1]]
                self.add_equ_x_equals(ca2,ending)

    def tell_time(self,seconds,togg_counter):
        """ Prints the approximated amount of time a print will take """
        total_mins = float((seconds+(togg_counter*.4))/60) # units: min
        hours = total_mins//60
        mins = 60*(total_mins/60 - hours)
        print('Approx printing time ~ ' + str(hours) + ' hours and ' + str(mins) + ' mins.')
    
    def t4(self):
        """ PRINT WITH TANGENTIAL ALIGNMENT USING ALGORITHM 4 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1
        
        # Calculate inc_x
        inc_x = self.nozzle_size*self.spacing # amount to increment x by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_x
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to draw each triangle and keeps running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Delete duplicates in list
        coords_list_no_dup = []
        for x in coords_list:
            if x not in coords_list_no_dup and x[0]<= 0: # so that we just get left of y-axis CUTS OBJECT IN HALF LENGTHWISE
                coords_list_no_dup.append(x)
        coords_arr = np.array(coords_list_no_dup)  # transform to a numpy array 
        # Sort by y coordinate
        #coords_arr_sorted = coords_arr[np.argsort(coords_arr[:, 1])]

        # Sort by x coordinate
        coords_arr_sorted = coords_arr[np.argsort(coords_arr[:,0])] 

        # Sort the half of the coordinates by distance from previous coordinate
        coords_list_sorted = coords_arr_sorted.tolist()
        starting = coords_list_sorted[0] # we want to start from a coordinate w the lowest y and among those, lowest x
        starting_ind = 0
        for i in range(len(coords_list_sorted)-1):
            if coords_list_sorted[i][1] < starting[1]:
                starting = coords_list_sorted[i]
                starting_ind = i
            if coords_list_sorted[i][1] == starting[1]:
                if coords_list_sorted[i][0] < starting[0]:
                    starting = coords_list_sorted[i]
                    starting_ind = i
        coord = starting # initialize coord to be the first set in the list
        others = np.delete(coords_list_sorted, starting_ind, 0)
        in_order = [coord] # initialize list that will have the coordinates in the right order **later convert to np array

        while True:
            if len(others) == 0:
                break
            else: # apply find_closest which returns index of closest set, then delete this item from others
                closest = self.find_closest(coord,others) # index
                in_order.append(others[closest])
                coord=others[closest]
                others = np.delete(others,closest,0)

        in_order_arr_left = np.array(in_order) # transform to numpy array

        highest_x = in_order_arr_left[0][0]
        highest_negative_y = in_order_arr_left[0][1]
        starting_coord_ind=0
        for i in range(len(in_order_arr_left)):
            if in_order_arr_left[i][0] >= highest_x:
                if in_order_arr_left[i][1] <= 0 and in_order_arr_left[i][1] > highest_negative_y: # makes sure y coordinate is negative 
                    highest_x = in_order_arr_left[i][0]
                    starting_coord_ind = i # this is the index of where we want to start printing til end then wrap around

        if self.is_circle == True:
            in_order_arr_left = np.delete(in_order_arr_left,len(in_order_arr_left)-1,0) # gets rid of center coord

        # rearrange in_order_arr_left so that we start at index i -> end and wrap around to hit the rest of the array
        rest = in_order_arr_left[:starting_coord_ind]
        in_order_arr_left_s = in_order_arr_left[starting_coord_ind::] # from starting point to end of arr
        in_order_arr_left =  np.concatenate((in_order_arr_left_s, rest), axis=0)

        # Now iterate over in_order_arr_left to calculate each line for left half
        self.line_dict = {} 
        a = in_order_arr_left[0][0]
        b = in_order_arr_left[0][1]
        first_coord_left = [a,b]
        c = in_order_arr_left[-1][0]
        d = in_order_arr_left[-1][1]
        last_coord_left = [c,d]
        for i in range(len(in_order_arr_left)-1):
            ca1 = in_order_arr_left[i]
            ca2 = in_order_arr_left[i+1]
            self.add_equ_y_equals(ca1, ca2)

        # MIRRORED Y: mirror left half over y-axis
        in_order_arr_right = in_order_arr_left
        in_order_arr_right[:,0] *= (-1)
        in_order_arr_right = in_order_arr_right[::-1]

        e = in_order_arr_right[0][0]
        f = in_order_arr_right[0][1]

        first_coord_right = [e,f]
        gg = in_order_arr_right[-1][0]
        h = in_order_arr_right[-1][1]
        last_coord_right = [gg,h]
        self.add_equ_y_equals(last_coord_left,first_coord_right) # connects left side to right

        for i in range(len(in_order_arr_right)-1):
            ca1 = in_order_arr_right[i]
            ca2 = in_order_arr_right[i+1]
            self.add_equ_y_equals(ca1, ca2)

        self.add_equ_y_equals(last_coord_right,first_coord_left) # connects last coord from right to first coord on left (closes shape)

        # MEANDER TIME
        #self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        lowest_x = coords_list_no_dup[0][0]
        lowest_coord = coords_list_no_dup[0]
        for c in coords_list_no_dup:
            if c[0] < lowest_x:
                lowest_x = c[0]
                lowest_coord = c # we will start our meander here (LEFT-MOST point so we can increment x to the right)
            if c[0] == lowest_x:
                if c[1] < lowest_coord[1]:
                    lowest_coord = c # makes sure we start at bottom left, not top left

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x =  lowest_coord[0]
            curr_y =  lowest_coord[1]
            curr_z = n*inc_x # range 0,1,2,etc. multiply inc_x for new z-height ## ONLY FOR ABSOLUTE MODE. INSTEAD USE REL
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # 1 for up, -1 for down
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_x) # relative move in z direction
            self.abs_move(curr_x,curr_y) # moving to starting point without actually printing

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1
            while True:
                if curr_x == lowest_coord[0]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                curr_x = round(curr_x,5)
                for key in self.line_dict: # goes through the dictionary to find [m b] lines in an appropriate x-range
                    if curr_x >= key[0] and curr_x <= key[1]: # checking if curr_x is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_y = []
                for l in possible_lines: # gets rid of any undefined (vertical) lines
                    for i in range(len(l)):
                        if l[i][0] != float('inf') and l[i][0] != float('-inf'):
                            m = l[i][0]
                            if l[i][1] != float('inf') and l[i][1] != float('-inf'):
                                b = l[i][1]
                                y = m*curr_x + b
                                if y not in possible_y:
                                    possible_y.append(y)

                if direction == -1:
                    possible_y.sort(reverse=True) # sort numbers in decreasing order since we want to print downwards
                else:
                    possible_y.sort(reverse=False) # sort numbers in increasing order since we want to print upwards

                for i in range(len(possible_y)): # now print a vertical line to each (curr_x,y) pair, updating curr_y
                    prev_y = curr_y # keep track of previous y -- used when calculating total time
                    curr_y = possible_y[i]
                    dist_y = abs(prev_y - curr_y) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_y)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_y/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all verticals together

                curr_x += inc_x # increment x to go toward the right
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction
            
            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)
        
        self.home()
        
        #self.view()
        #self.teardown()

    def l4(self, xs,ys):
        """ PRINT WITH LONGITUDINAL ALIGNMENT USING ALGORITHM 4 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1

        # Calculate inc_y
        inc_y = self.nozzle_size*self.spacing # amount to increment y by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_y
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to draw each triangle and keeps running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Delete duplicates in list
        coords_list_no_dup = []
        for x in coords_list:
            if x not in coords_list_no_dup and x[0]<= 0: # so that we just get left of y-axis CUTS OBJECT IN HALF LENGTHWISE:
                coords_list_no_dup.append(x)
        coords_arr = np.array(coords_list_no_dup)  # transform to a numpy array 

        # Sort by x coordinate
        coords_arr_sorted = coords_arr[np.argsort(coords_arr[:,0])] 

        # Sort the half of the coordinates by distance from previous coordinate
        coords_list_sorted = coords_arr_sorted.tolist()
        starting = coords_list_sorted[0] # we want to start from a coordinate w the lowest y and among those, lowest x
        starting_ind = 0
        for i in range(len(coords_list_sorted)-1):
            if coords_list_sorted[i][1] < starting[1]:
                starting = coords_list_sorted[i]
                starting_ind = i
            if coords_list_sorted[i][1] == starting[1]:
                if coords_list_sorted[i][0] < starting[0]:
                    starting = coords_list_sorted[i]
                    starting_ind = i
        coord = starting # initialize coord to be the first set in the list
        others = np.delete(coords_list_sorted, starting_ind, 0)
        in_order = [coord] # initialize list that will have the coordinates in the right order **later convert to np array

        while True:
            if len(others) == 0:
                break
            else: # apply find_closest which returns index of closest set, then delete this item from others
                closest = self.find_closest(coord,others) # index
                in_order.append(others[closest])
                coord=others[closest]
                others = np.delete(others,closest,0)

        in_order_arr_left = np.array(in_order) # transform to numpy array *** whole left side not just top left

        highest_x = in_order_arr_left[0][0]
        highest_negative_y = in_order_arr_left[0][1]
        starting_coord_ind = 0

        for i in range(len(in_order_arr_left)):
            if in_order_arr_left[i][0] >= highest_x:
                if in_order_arr_left[i][1] <= 0 and in_order_arr_left[i][1] > highest_negative_y: # makes sure y coordinate is negative 
                    highest_x = in_order_arr_left[i][0]
                    starting_coord_ind = i # this is the index of where we want to start printing til end then wrap around

        if self.is_circle == True:
            in_order_arr_left = np.delete(in_order_arr_left,len(in_order_arr_left)-1,0) # gets rid of center coord

        # rearrange in_order_arr_left so that we start at index i -> end and wrap around to hit the rest of the array
        rest = in_order_arr_left[:starting_coord_ind]
        in_order_arr_left_s = in_order_arr_left[starting_coord_ind::] # from starting point to end of arr
        in_order_arr_left =  np.concatenate((in_order_arr_left_s, rest), axis=0)

        # Now iterate over in_order_arr_left to calculate each line for left half
        self.line_dict = {}
        a = in_order_arr_left[0][0]
        b = in_order_arr_left[0][1]
        first_coord_left = [a,b]
        c = in_order_arr_left[-1][0]
        d = in_order_arr_left[-1][1]
        last_coord_left = [c,d]
        for i in range(len(in_order_arr_left)-1):
            ca1 = in_order_arr_left[i]
            ca2 = in_order_arr_left[i+1]
            self.add_equ_x_equals(ca1, ca2)

        # MIRRORED Y: mirror left half over y-axis
        in_order_arr_right = in_order_arr_left
        in_order_arr_right[:,0] *= (-1)
        in_order_arr_right = in_order_arr_right[::-1]

        e = in_order_arr_right[0][0]
        f = in_order_arr_right[0][1]

        first_coord_right = [e,f]
        gg = in_order_arr_right[-1][0]
        h = in_order_arr_right[-1][1]
        last_coord_right = [gg,h]
        self.add_equ_x_equals(last_coord_left,first_coord_right) # connects left side to right

        for i in range(len(in_order_arr_right)-1):
            ca1 = in_order_arr_right[i]
            ca2 = in_order_arr_right[i+1]
            self.add_equ_x_equals(ca1, ca2)

        self.add_equ_x_equals(last_coord_right,first_coord_left) # connects last coord from right to first coord on left (closes shape)

        # for key, value in self.line_dict.items():
        #     print(key, ' : ', value)
        
        # MEANDER TIME
        #self.set_home(0,0) # Ramon commented this
        seconds = 0
        togg_counter = 0

        highest_y = coords_list_no_dup[0][1]
        highest_coord = coords_list_no_dup[0]
        for c in coords_list_no_dup:
            if c[1] > highest_y:
                highest_y = c[1]
                highest_coord = c # we will start our meander here (highest point so we can increment y down)
            if c[1] == highest_y:
                if c[0] < highest_coord[0]: #checking for highest coord w lowest x val (top left starting point)
                    highest_coord = c

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x =  highest_coord[0]
            curr_y =  highest_coord[1]
            curr_z = n*inc_y # range 0,1,2,etc. will multiply inc_x for new z-height
            print('curr_x: ' +str(curr_x) + '. curr_y: '+str(curr_y))
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # 1 for right, -1 for left
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z = inc_y)
            self.abs_move(curr_x+xs,curr_y+ys) # Ramon changed this

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1
            
            while True:
                if curr_y == highest_coord[1]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                curr_y = round(curr_y,5)
                for key in self.line_dict: # goes through the dictionary to find [m b] or [x] lines in an appropriate y-range
                    if curr_y >= key[0] and curr_y <= key[1]: # checking if curr_y is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] or [x] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_x = []
                for l in possible_lines: 
                    for i in range(len(l)):
                        if len(l[i]) == 1:
                            x = l[i][0]
                            if x not in possible_x:
                                possible_x.append(x) 
                        else:
                            m = l[i][0]
                            b = l[i][1]
                            if m != 0:
                                x = (curr_y-b) / m
                                if x not in possible_x:
                                    possible_x.append(x)    
                if direction == -1:
                    possible_x.sort(reverse=True) # sort numbers in decreasing order since we want to print traveling left
                else:
                    possible_x.sort(reverse=False) # sort numbers in increasing order since we want to print traveling right
                
                for i in range(len(possible_x)): # now print a horizontal/longit line to each (curr_x,y) pair, updatng curr_x
                    prev_x = curr_x + xs # Ramon changed this
                    curr_x = possible_x[i]
                    dist_x = abs(prev_x - curr_x) # for calculating total time
                    self.absolute()
                    self.move(curr_x+xs,curr_y+ys) # Ramon changed this
                    if i != 0 and i != (len(possible_x)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_x/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all printed lines together
                
                curr_y -= inc_y # increment y so that we move downwards
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)

        #self.home() #Beibit: commented, for my printing it is effective without homing
        #self.teardown()

    def t2(self):
        """ PRINT WITH TANGENTIAL ALIGNMENT USING ALGORITHM 2 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1

        # Calculate inc_x
        inc_x = self.nozzle_size*self.spacing # amount to increment x by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_x
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to keep a running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Sort and remove duplicate coordinates... not so efficient but dictionaries didn't work
        coords_list.sort()

        coords_list2 = []
        for x in coords_list:
            x[0] = round(x[0],10)
            x[1] = round(x[1],10)
            if x not in coords_list2 and self.is_circle == False:
                coords_list2.append(x)
            elif x not in coords_list2 and self.is_circle == True and x!=[0,0]:
                coords_list2.append(x)
        coords_arr = np.array(coords_list2)  # transform to a numpy array of all the coords

        # Finding polar angle theta of each coord and storing in a vertical array -- preserves same indices of coords_arr
        # Uses theta function
        theta_arr = np.empty([len(coords_arr),1])
        for r in range(len(coords_arr)): #start iterating at the second row since first is taken care of
            x = coords_arr[r][0]
            y = coords_arr[r][1]
            t = self.calc_theta(x,y)
            theta_arr[r,0] = t

        # Add theta_arr as third column of coords_arr and sort coords_arr based on this third column
        combined_arr = np.hstack((coords_arr,theta_arr))
        coords_sorted = combined_arr[np.argsort(combined_arr[:,2])]
        coords_sorted = np.delete(coords_sorted, 2, axis=1) # leaves just the coordinates in numpy array

        # Now iterate over in_order_arr and calculate equation of lines btwn points and add to line_dict dictionary
        self.line_dict = {}
        for i in range(len(coords_sorted)-1):
            ca1 = coords_sorted[i]
            ca2 = coords_sorted[i+1]
            self.add_equ_y_equals(ca1, ca2)
        self.add_equ_y_equals(coords_sorted[0],coords_sorted[len(coords_sorted)-1])

        """
        for key, value in line_dict.items():
            print(key, ' : ', value)
        """

        # MEANDER TIME
        self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        lowest_x = coords_sorted[0][0]
        lowest_coord = coords_sorted[0]
        for c in coords_sorted:
            if c[0] < lowest_x:
                lowest_x = c[0]
                lowest_coord = c # we will start our meander here (left-most point so we can increment x to the right)

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x = lowest_coord[0] # starting print at first coordinate
            curr_y = lowest_coord[1]
            curr_z = n*inc_x # range 0,1,2,etc. will multiply inc_x for new z-height ## ONLY FOR ABS MODE. USE REL INSTEAD!!
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # -1 for down, 1 for up
            possible_lines = []
                    
            if n !=0:
                self.relative()
                self.move(z=inc_x)
            self.abs_move(curr_x,curr_y) # not printing yet, just moving to starting position

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1

            while True:
                if curr_x == lowest_coord[0]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] lines in an appropriate x-range
                    if curr_x >= key[0] and curr_x <= key[1]: # checking if curr_x is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_y = []
                for l in possible_lines: # gets rid of any undefined (vertical/tangential) lines as well as duplicate y vals
                    for i in range(len(l)):
                        if l[i][0] != float('inf') and l[i][0] != float('-inf'):
                            m = l[i][0]
                            if l[i][1] != float('inf') and l[i][1] != float('-inf'):
                                b = l[i][1]
                                y = m*curr_x + b
                                if y not in possible_y: #and y != curr_y:
                                    possible_y.append(y)

                if direction == -1:
                    possible_y.sort(reverse=True) # sort numbers in decreasing order since we want to print downwards
                else:
                    possible_y.sort(reverse=False) # sort numbers in increasing order since we want to print upwards

                for i in range(len(possible_y)): # now print a vertical/tangential line to each (curr_x,y) pair, updating curr_y
                    prev_y = curr_y # keep track of previous y -- used when calculating total time
                    curr_y = possible_y[i]
                    dist_y = abs(prev_y - curr_y) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_y)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_y/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all printed lines together

                curr_x += inc_x # increment x
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)

        self.home() # move back to home position
        
        #self.teardown()

    def l2(self, xs, ys):
        """ PRINT WITH LONGITUDINAL ALIGNMENT USING ALGORITHM 2 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1

        # Calculate inc_y
        inc_y = self.nozzle_size*self.spacing # amount to increment y by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_y
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to keep a running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Sort and remove duplicate coordinates... not so efficient but dictionaries didn't work
        coords_list.sort()

        coords_list2 = []
        for x in coords_list:
            x[0] = round(x[0],10)
            x[1] = round(x[1],10)
            if x not in coords_list2 and self.is_circle == False:
                coords_list2.append(x)
            elif x not in coords_list2 and self.is_circle == True and x!=[0,0]:
                coords_list2.append(x)
        coords_arr = np.array(coords_list2)  # transform to a numpy array of all the coords

        # Finding polar angle theta of each coord and storing in a vertical array -- preserves same indices of coords_arr
        # Uses calc_theta() function
        theta_arr = np.empty([len(coords_arr),1])
        for r in range(len(coords_arr)): #start iterating at the second row since first is taken care of
            x = coords_arr[r][0]
            y = coords_arr[r][1]
            t = self.calc_theta(x,y)
            theta_arr[r,0] = t

        # Add theta_arr as third column of coords_arr and sort coords_arr based on this third column
        combined_arr = np.hstack((coords_arr,theta_arr))
        coords_sorted = combined_arr[np.argsort(combined_arr[:,2])]
        coords_sorted = np.delete(coords_sorted, 2, axis=1) # leaves just the coordinates in numpy array

        # Now iterate over in_order_arr and calculate equation of lines btwn points and add to line_dict dictionary
        self.line_dict = {}
        for i in range(len(coords_sorted)-1):
            ca1 = coords_sorted[i]
            ca2 = coords_sorted[i+1]
            self.add_equ_x_equals(ca1, ca2)
        self.add_equ_x_equals(coords_sorted[0],coords_sorted[len(coords_sorted)-1])

        #for key, value in line_dict.items():
        #    print(key, ' : ', value)

        # MEANDER TIME
        #self.set_home(0,0) # Ramon : Commented this
        seconds = 0
        togg_counter = 0

        highest_y = coords_sorted[0][1]
        highest_coord = coords_sorted[0]
        for c in coords_sorted:
            if c[1] > highest_y:
                highest_y = c[1]
                highest_coord = c # we will start our meander here (left-most point so we can increment x to the right)

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x = highest_coord[0] # starting print at first coordinate
            curr_y = highest_coord[1]
            curr_z = n*inc_y # range 0,1,2,etc. will multiply inc_x for new z-height ## ONLY FOR ABS MODE. INSTEAD USE REL!
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # -1 for left, 1 for right
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_y)
            self.abs_move(curr_x+xs,curr_y+ys) # # Ramon: I added xs and ys so that its not always starting from 0,0, delete it if you don't need it

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1

            while True:
                if curr_y == highest_coord[1]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] or [x] lines in an appropriate y-range
                    if curr_y >= key[0] and curr_y <= key[1]: # checking if curr_y is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] or [x] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_x = []
                for l in possible_lines: 
                    for i in range(len(l)):
                        if len(l[i]) == 1:
                            x = l[i][0]
                            x = round(x,9)
                            if x not in possible_x:
                                possible_x.append(x) 
                        else:
                            m = l[i][0]
                            b = l[i][1]
                            if m != 0:
                                x = (curr_y-b) / m
                                x = round(x,9)
                                if x not in possible_x:
                                    possible_x.append(x)    
                if direction == -1:
                    possible_x.sort(reverse=True) # sort numbers in decreasing order since we want to print traveling left
                else:
                    possible_x.sort(reverse=False) # sort numbers in increasing order since we want to print traveling right
            
                for i in range(len(possible_x)): # now print a horizontal/longit line to each (curr_x,y) pair, updatng curr_x
                    prev_x = curr_x
                    curr_x = possible_x[i]
                    dist_x = abs(prev_x - curr_x) # for calculating total time
                    self.absolute()
                    self.move(curr_x+xs,curr_y+ys) # Ramon: I added xs and ys so that its not always starting from 0,0, delete it if you don't need it
                    if i != 0 and i != (len(possible_x)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_x/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all printed lines together
                    
                curr_y -= inc_y # increment y
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)
        self.abs_move(z=-80) # Ramon added this
        self.home()
        #self.teardown()

    def t1(self):
        """ PRINT WITH TANGENTIAL ALIGNMENT USING ALGORITHM 1 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1

        # Calculate inc_x
        inc_x = self.nozzle_size*self.spacing # amount to increment x by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_x
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to draw each triangle and keeps running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Sort and remove duplicate coordinates... not so efficient but dictionaries didn't work
        coords_list.sort()

        coords_list2 = []
        for x in coords_list:
            if x not in coords_list2:
                coords_list2.append(x)

        x = round(coords_list2[0][0],10)
        y = round(coords_list2[0][1],10)
        coord = [x,y] # initialize coord to be the first set in the list
        others = np.delete(coords_list2,0,0) # initialize others by deleting first coord from coords_arr array
        in_order = [coord] # initialize list that will have the coordinates in the right order **later convert to np array
        while True:
            if len(others) == 0:
                break
            else: # apply find_closest which returns index of closest set, then delete this item from others
                closest = self.find_closest(coord,others) # index
                coord = others[closest]
                x = round(coord[0],10)
                y = round(coord[1],10)
                coord = [x,y]
                in_order.append(coord)
                others = np.delete(others,closest,0)

        in_order_arr = np.array(in_order)

        if self.is_circle == True:
            in_order_arr = np.delete(in_order_arr,len(in_order_arr)-1,0) # gets rid of center coord

        # Now iterate over in_order_arr to calculate each line
        self.line_dict = {}
        for i in range(len(in_order_arr)-1):
            ca1 = in_order_arr[i]
            ca2 = in_order_arr[i+1]
            self.add_equ_y_equals(ca1, ca2)

        self.add_equ_y_equals(in_order_arr[0],in_order_arr[len(in_order_arr)-1])

        # MEANDER TIME
        self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        lowest_x = coords_list2[0][0]
        lowest_coord = coords_list2[0]
        for c in coords_list2:
            if c[0] < lowest_x:
                lowest_x = c[0]
                lowest_coord = c # we will start our meander here (left-most point so we can increment x up and right)

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x = lowest_coord[0] # starting print at first coordinate
            curr_y = lowest_coord[1]
            curr_z = n*inc_x # range 0,1,2,etc. will multiply inc_x for new z-height
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # -1 for down, 1 for up
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_x)
            self.abs_move(curr_x,curr_y) # not printing yet, just moving to starting position @ NEW Z-HEIGHT

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1

            while True:
                if curr_x == lowest_coord[0]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] lines in an appropriate x-range
                    if curr_x >= key[0] and curr_x <= key[1]: # checking if curr_x is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_y = []
                for l in possible_lines: # gets rid of any undefined (vertical/tangential) lines as well as duplicate y vals
                    for i in range(len(l)):
                        if l[i][0] != float('inf') and l[i][0] != float('-inf'):
                            m = l[i][0]
                            if l[i][1] != float('inf') and l[i][1] != float('-inf'):
                                b = l[i][1]
                                y = m*curr_x + b
                                if y not in possible_y:
                                    possible_y.append(y)

                if direction == -1:
                    possible_y.sort(reverse=True) # sort numbers in decreasing order since we want to print downwards
                else:
                    possible_y.sort(reverse=False) # sort numbers in increasing order since we want to print upwards
                    
                for i in range(len(possible_y)): # now print a vertical/tangential line to each (curr_x,y) pair, updating curr_y
                    prev_y = curr_y # keep track of previous y -- used when calculating total time
                    curr_y = possible_y[i]
                    dist_y = abs(prev_y - curr_y) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_y)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_y/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all printed lines together

                curr_x += inc_x # increment x
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)

        self.home()
        #self.teardown()

    def l1(self):
        """ PRINT WITH LONGITUDINAL ALIGNMENT USING ALGORITHM 1 """
        # Load the STL file and create array
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        m_array = np.array(my_mesh) #creates a numpy array from the STL file --> represents coords of each triangle

        row_num = 0
        rows_to_delete=[]
        z_height = 0 # we are going to find the max z -> this is the height so we know how many layers to print
        for r in m_array: # loops through each row to create a list of indexes of rows to be deleted (ones w/ z-coords)
            if r[2] != 0 or r[5]!=0 or r[8]!=0:
                rows_to_delete.append(row_num)
            if r[2] > z_height or r[5] > z_height or r[8] > z_height:
                z_height = max(r[2], r[5], r[8])
            row_num+=1

        # Calculate inc_y
        inc_y = self.nozzle_size*self.spacing # amount to increment y by during "meander"

        # Calculate number of layers to print from STL file
        if self.num_layers == None:
            self.num_layers = z_height / inc_y
            self.num_layers = int(round(self.num_layers))# rounds up or down (whichever is closer)

        # Delete the designated rows
        base_array = np.delete(m_array,rows_to_delete,0)

        # Create empty list to store coordinate pairs
        coords_list = []

        for r in base_array: # loops through each row to draw each triangle and keeps running list of coords
            x1 = r[0]
            y1 = r[1]
            a=[x1,y1]
            x2 = r[3]
            y2 = r[4]
            b=[x2,y2]
            x3 = r[6]
            y3 = r[7]
            c=[x3,y3]
            # now add the coords to the master list
            coords_list.append(a)
            coords_list.append(b)
            coords_list.append(c)

        # Sort and remove duplicate coordinates... not so efficient but dictionaries didn't work
        coords_list.sort()

        coords_list2 = []
        for x in coords_list:
            if x not in coords_list2:
                coords_list2.append(x)

        x = round(coords_list2[0][0],10)
        y = round(coords_list2[0][1],10)
        coord = [x,y] # initialize coord to be the first set in the list
        others = np.delete(coords_list2,0,0) # initialize others by deleting first coord from coords_arr array
        in_order = [coord] # initialize list that will have the coordinates in the right order **later convert to np array
        while True:
            if len(others) == 0:
                break
            else: # apply find_closest which returns index of closest set, then delete this item from others
                closest = self.find_closest(coord,others) # index
                coord = others[closest]
                x = round(coord[0],10)
                y = round(coord[1],10)
                coord = [x,y]
                in_order.append(coord)
                others = np.delete(others,closest,0)

        in_order_arr = np.array(in_order)

        if self.is_circle == True:
            in_order_arr = np.delete(in_order_arr,len(in_order_arr)-1,0)

        # Now iterate over in_order_arr to calculate each line
        self.line_dict = {}
        for i in range(len(in_order_arr)-1):
            ca1 = in_order_arr[i]
            ca2 = in_order_arr[i+1]
            self.add_equ_x_equals(ca1, ca2)

        self.add_equ_x_equals(in_order_arr[0],in_order_arr[len(in_order_arr)-1])

        # MEANDER TIME
        self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        highest_y = coords_list2[0][1]
        highest_coord = coords_list2[0]
        for c in coords_list2:
            if c[1] > highest_y:
                highest_y = c[1]
                highest_coord = c # we will start our meander here (highest point so we can increment y down)

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x =  highest_coord[0] #in_order_arr[2][0] # starting print in top left corner 
            curr_y =  highest_coord[1] #in_order_arr[2][1] 
            curr_z = n*inc_y # range 0,1,2,etc. will multiply inc_y for new z-height
            print('The current z position is ' + str(curr_z) + ' mm.') 
            direction = 1 # 1 for right, -1 for left
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_y)
            self.abs_move(curr_x,curr_y) # not printing yet, just moving to starting position

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1

            while True:
                if curr_y == highest_coord[1]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] or [x] lines in an appropriate y-range
                    if curr_y >= key[0] and curr_y <= key[1]: # checking if curr_y is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] or [x] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_x = []
                for l in possible_lines: 
                    for i in range(len(l)):
                        if len(l[i]) == 1:
                            x = l[i][0]
                            if x not in possible_x:
                                possible_x.append(x) 
                        else:
                            m = l[i][0]
                            b = l[i][1]
                            if m != 0:
                                x = (curr_y-b) / m
                                if x not in possible_x:
                                    possible_x.append(x)    
                if direction == -1:
                    possible_x.sort(reverse=True) # sort numbers in decreasing order since we want to print traveling left
                else:
                    possible_x.sort(reverse=False) # sort numbers in increasing order since we want to print traveling right
                
                for i in range(len(possible_x)): # now print a horizontal/longit line to each (curr_x,y) pair, updatng curr_x
                    prev_x = curr_x
                    curr_x = possible_x[i]
                    dist_x = abs(prev_x - curr_x) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_x)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    seconds += (dist_x/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all printed lines together
                
                curr_y -= inc_y # increment y
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)

        self.tell_time(seconds,togg_counter)

        self.home()    
        #self.teardown()
    
    def gt(self):
        """ Input file of g-code (.pgm)
            For arcs, make sure to use the radius method (not IJK)
            TANGENTIAL alignment        
        """
        shape_file = open(str(self.pgm_file), "r")
        relavant_commands = ['G0 ','G1 ','G2 ','G3 ','G90','G91']
        togg = -1 # -1 is off, 1 is on

        # Calculate inc_x
        inc_x = self.nozzle_size*self.spacing # amount to increment x by during "meander"

        g_lines = []
        for x in shape_file:
            if (x[0:3] in relavant_commands or x[0:11] == 'Call toggle') and 'F' not in x:
                g_lines += [x]
        g_lines = [x.strip('\n') for x in g_lines] # gets rid of extra line space

        # Use the lines of g-code to determine coordinates and calc equation w previous coord to add to line_dict
        self.line_dict={}
        prev_x = 0
        prev_y = 0
        curr_x = 0
        curr_y = 0
        for l in g_lines:
            if l[0:3] == 'G90': # absolute mode
                mode = 'a'
            elif l[0:3] == 'G91': # relative mode
                mode = 'r'
            elif l[0:11] == 'Call toggle':
                togg *= -1 # changes on/off back and forth
            elif l[0:2] == 'G0' or l[0:2] == 'G1': # rapid move or normal move (both get to the same place)
                x_ind = l.find('X') +1
                y_ind = l.find('Y') +1

                if x_ind != 0:
                    x1 = float(l[x_ind:y_ind-2])
                else:
                    x1 = 0
                if y_ind != 0:
                    y1 = float(l[y_ind:y_ind+8])
                else:
                    y1 = 0

                if togg == -1: # not printing yet, just updating position
                    if mode == 'a': # absolute mode; the coordinates are the given numbers
                        prev_x = x1
                        prev_y = y1
                    else: # relative mode; add to previous value
                        prev_x += x1
                        prev_y += y1
                else: # togg == 1
                    if mode == 'a': # absolute
                        curr_x = x1
                        curr_y = y1
                    else: # relative
                        curr_x = prev_x + x1
                        curr_y = prev_y +y1
                    
                    ca1 = np.array([prev_x,prev_y])
                    ca2 = np.array([curr_x,curr_y])
                    self.add_equ_y_equals(ca1,ca2)

                    prev_x = curr_x
                    prev_y = curr_y
            else: # G2 or G3 - CW or CCW arcs
                if l[0:2] == "G2":
                    direction = 1
                else:
                    direction = -1

                x_ind = l.find('X') +1
                y_ind = l.find('Y') +1
                r_ind = l.find('R') +1
                radius = float(l[r_ind::])

                if mode == 'a': # absolute
                    if x_ind != 0:
                        x1 = float(l[x_ind:y_ind-2])
                    else:
                        x1 = 0
                    if y_ind != 0:
                        y1 = float(l[y_ind:y_ind+8])
                    else:
                        y1 = 0
                else: # relative
                    if x_ind != 0:
                        x1 = prev_x + float(l[x_ind:y_ind-2])
                    else:
                        x1 = prev_x
                    if y_ind != 0:
                        y1 = prev_y + float(l[y_ind:y_ind+8])
                    else:
                        y1 = prev_y

                end_pt = [x1,y1]
                start_pt = [prev_x,prev_y] # previous pt
                center_pt = [((x1+prev_x)/2),((y1+prev_y)/2)]

                self.g_arc_t(center_pt,start_pt,end_pt,direction,radius,resolution=40)
                prev_x=x1
                prev_y=y1
                
        # print('line_dict:')
        # for key, value in line_dict.items():
        #     print(key, ' : ', value)

        keys = self.line_dict.keys()
        values = self.line_dict.values()
        lowest_x = keys[0][0]
        m = self.line_dict[keys[0]][0][0]
        b = self.line_dict[keys[0]][0][1]
        lowest_y = m*lowest_x + b
        for i in range(len(keys)):
            if keys[i][0] <= lowest_x:
                for v in self.line_dict[keys[i]]:
                    m = v[0]
                    b = v[1]
                    y = m*keys[i][0] + b
                    if y < lowest_y:
                        lowest_x = keys[i][0]
                        lowest_y = y

        # MEANDER TIME
        self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x =  lowest_x
            curr_y =  lowest_y
            curr_z = n*inc_x # range 0,1,2,etc. multiply inc_x for new z-height ## ONLY FOR ABSOLUTE MODE. INSTEAD USE REL
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # 1 for up, -1 for down
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_x) # relative move in z direction

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1

            self.abs_move(curr_x,curr_y) # moving to starting point without actually printing
            while True:
                if curr_x == lowest_x and curr_y == lowest_y:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] lines in an appropriate x-range
                    if curr_x >= key[0] and curr_x <= key[1]: # checking if curr_x is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_y = []
                for l in possible_lines: # gets rid of any undefined (vertical) lines
                    for i in range(len(l)):
                        if l[i][0] != float('inf') and l[i][0] != float('-inf'):
                            m = l[i][0]
                            if l[i][1] != float('inf') and l[i][1] != float('-inf'):
                                b = l[i][1]
                                y = m*curr_x + b
                                if y not in possible_y:
                                    possible_y.append(y)

                if direction == -1:
                    possible_y.sort(reverse=True) # sort numbers in decreasing order since we want to print downwards
                else:
                    possible_y.sort(reverse=False) # sort numbers in increasing order since we want to print upwards

                for i in range(len(possible_y)): # now print a vertical line to each (curr_x,y) pair, updating curr_y
                    prev_y = curr_y # keep track of previous y -- used when calculating total time
                    curr_y = possible_y[i]
                    dist_y = abs(prev_y - curr_y) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_y)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_y/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all verticals together

                curr_x += inc_x # increment x to go toward the right
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)

        self.home()
        #self.teardown()

    def gl(self):
        """ Input file of g-code (.pgm)
            For arcs, make sure to use the radius method (not IJK)
            LONGITUDINAL alignment        
        """
        shape_file = open(str(self.pgm_file), "r")
        relavant_commands = ['G0 ','G1 ','G2 ','G3 ','G90','G91']
        togg = -1 # -1 is off, 1 is on

        # Calculate inc_y
        inc_y = self.nozzle_size*self.spacing # amount to increment x by during "meander"

        g_lines = []
        for x in shape_file:
            if (x[0:3] in relavant_commands or x[0:11] == 'Call toggle') and 'F' not in x:
                g_lines += [x]
        g_lines = [x.strip('\n') for x in g_lines] # gets rid of extra line space

        # Use the lines of g-code to determine coordinates and calc equation w previous coord to add to line_dict
        self.line_dict={}
        prev_x = 0
        prev_y = 0
        curr_x = 0
        curr_y = 0
        list_of_coords=[]
        for l in g_lines:
            if l[0:3] == 'G90': # absolute mode
                mode = 'a'
            elif l[0:3] == 'G91': # relative mode
                mode = 'r'
            elif l[0:11] == 'Call toggle':
                togg *= -1 # changes on/off back and forth
            elif l[0:2] == 'G0' or l[0:2] == 'G1': # rapid move or normal move (both get to the same place)
                x_ind = l.find('X') +1
                y_ind = l.find('Y') +1

                if x_ind != 0:
                    x1 = float(l[x_ind:y_ind-2])
                else:
                    x1 = 0
                if y_ind != 0:
                    y1 = float(l[y_ind:y_ind+8])
                else:
                    y1 = 0

                if togg == -1: # not printing yet, just updating position
                    if mode == 'a': # absolute mode; the coordinates are the given numbers
                        prev_x = x1
                        prev_y = y1
                    else: # relative mode; add to previous value
                        prev_x += x1
                        prev_y += y1
                else: # togg == 1
                    if mode == 'a': # absolute
                        curr_x = x1
                        curr_y = y1
                    else: # relative
                        curr_x = prev_x + x1
                        curr_y = prev_y + y1
                    
                    ca1 = np.array([prev_x,prev_y])
                    ca2 = np.array([curr_x,curr_y])
                    self.add_equ_x_equals(ca1,ca2)

                    prev_x = curr_x
                    prev_y = curr_y
            else: # G2 or G3 - CW or CCW arcs
                if l[0:2] == "G2":
                    direction = 1
                else:
                    direction = -1

                x_ind = l.find('X') +1
                y_ind = l.find('Y') +1
                r_ind = l.find('R') +1
                radius = float(l[r_ind::])

                if mode == 'a': # absolute
                    if x_ind != 0:
                        x1 = float(l[x_ind:y_ind-2])
                    else:
                        x1 = 0
                    if y_ind != 0:
                        y1 = float(l[y_ind:y_ind+8])
                    else:
                        y1 = 0
                else: # relative
                    if x_ind != 0:
                        x1 = prev_x + float(l[x_ind:y_ind-2])
                    else:
                        x1 = prev_x
                    if y_ind != 0:
                        y1 = prev_y + float(l[y_ind:y_ind+8])
                    else:
                        y1 = prev_y

                end_pt = [x1,y1]
                start_pt = [prev_x,prev_y] # previous pt
                center_pt = [((x1+prev_x)/2),((y1+prev_y)/2)]

                resolution = 40
                self.g_arc_l(center_pt,start_pt,end_pt,direction,radius,resolution)
                prev_x=x1
                prev_y=y1

        # print('line_dict:')
        # for key, value in line_dict.items():
        #     print(key, ' : ', value)

        keys = self.line_dict.keys()

        values = self.line_dict.values()

        highest_y = keys[0][1] # upper y range value of first entry in dictionary
        highest_coord_keys = [keys[0]] # keep track of then use m and b to find x

        for i in range(len(keys)):
            if keys[i][1] > highest_y:
                highest_y = keys[i][1]
                highest_coord_keys = [keys[i]] # we will start our meander here (highest point so we can increment y down)
            if keys[i][1] == highest_y and keys[i] not in highest_coord_keys:
                highest_coord_keys.append(keys[i])
        highest_coord = highest_coord_keys[0]
        highest_xs = []

        for h in highest_coord_keys:
            for v in self.line_dict[h]:
                if len(v) == 1:
                    x = v[0]
                    highest_xs.append(x)
                else:
                    m = v[0]
                    b = v[1]
                    if m != 0:
                        x = (highest_y-b) / m
                        highest_xs.append(x)

        highest_coord = [min(highest_xs),highest_y]

        # MEANDER TIME
        self.set_home(0,0)
        seconds = 0
        togg_counter = 0

        for n in range(self.num_layers): # repeat printing base shape 
            curr_x =  highest_coord[0]
            curr_y =  highest_coord[1]
            curr_z = n*inc_y # range 0,1,2,etc. multiply inc_y for new z-height ## ONLY FOR ABSOLUTE MODE. INSTEAD USE REL
            print('The current z position is ' + str(curr_z) + ' mm.')
            direction = 1 # 1 for up, -1 for down
            possible_lines = []

            if n != 0:
                self.relative()
                self.move(z=inc_y) # relative move in z direction

            if self.pressure_box == "Nordson":
                action_counter = 2
                action = "toggle"
            elif self.pressure_box == "Alicat":
                action_counter = -1
            else: # Solenoid
                action_counter = -1
                
            self.abs_move(curr_x,curr_y) # moving to starting point without actually printing
            while True:
                if curr_x == highest_coord[0] and curr_y == highest_coord[1]:
                    action_counter*=-1
                    action=self.action_switch(action_counter)
                    self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                    togg_counter += 1
                for key in self.line_dict: # goes through the dictionary to find [m b] lines in an appropriate x-range
                    if curr_y >= key[0] and curr_y <= key[1]: # checking if curr_y is in a key's range
                        possible_lines.append(self.line_dict[key]) # add the [m b] associated w that key
                if len(possible_lines) == 0:
                    break

                possible_x = []
                for l in possible_lines: 
                    for i in range(len(l)):
                        if len(l[i]) == 1:
                            x = l[i][0]
                            if x not in possible_x:
                                possible_x.append(x) 
                        else:
                            m = l[i][0]
                            b = l[i][1]
                            if m != 0:
                                x = (curr_y-b) / m
                                if x not in possible_x:
                                    possible_x.append(x)    
                if direction == -1:
                    possible_x.sort(reverse=True) # sort numbers in decreasing order since we want to print traveling left
                else:
                    possible_x.sort(reverse=False) # sort numbers in increasing order since we want to print traveling right
                
                for i in range(len(possible_x)): # now print a horizontal line to each (curr_x,y) pair, updatng curr_x
                    prev_x = curr_x
                    curr_x = possible_x[i]
                    dist_x = abs(prev_x - curr_x) # for calculating total time
                    self.absolute()
                    self.move(curr_x,curr_y)
                    if i != 0 and i != (len(possible_x)-1):
                        action_counter*=-1
                        action=self.action_switch(action_counter)
                        self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
                        togg_counter += 1
                    seconds += (dist_x/self.print_speed) # divide distance (mm) by speed (mm/s) = s --> add all verticals together
                
                curr_y -= inc_y # increment y so that we move downwards
                possible_lines = [] # reset possible lines to empty list
                direction *= -1 # switch print direction

            action_counter*=-1
            action=self.action_switch(action_counter)
            self.toggle_valve(self.pressure_box,action,self.com_port,self.valve_port)
            togg_counter += 1

        self.tell_time(seconds,togg_counter)

    
        self.home() 
       
        #self.teardown()

    def slicer(self,xs=0,ys=0): # Ramon: I added xs and ys so that its not always starting from 0,0, delete it if you don't need it
        if self.algorithm_num == 4:
            if self.alignment == 't':
                self.t4()
            elif self.alignment == 'l':
                self.l4(xs,ys)
            else:
                print('INPUT ERROR: please enter t for tangential alignment or l for longitudinal alignment of print')
        elif self.algorithm_num == 2:
            if self.alignment == 't':
                self.t2()
            elif self.alignment == 'l':
                self.l2(xs,ys) # Ramon: I added xs and ys so that its not always starting from 0,0, delete it if you don't need it
            elif self.alignment == 'f':
                self.f2()
            else:
                print('INPUT ERROR: please enter t for tangential alignment or l for longitudinal alignment of print')
        elif self.algorithm_num == 1:
            if self.alignment == 't':
                self.t1()
            elif self.alignment == 'l':
                self.l1()
            elif self.alignment == 'f':
                self.f1()
            else:
                print('INPUT ERROR: please enter t for tangential alignment or l for longitudinal alignment of print')
        elif self.algorithm_num == 3:
            if self.alignment == 'f':
                self.f3()
        elif self.algorithm_num == None: # g code time
            if self.alignment == 't':
                self.gt()
            elif self.alignment == 'l':
                self.gl()
            else:
                print('INPUT ERROR: please enter t for tangential alignment or l for longitudinal alignment of print')
        else:
            print('INPUT ERROR: enter algorithm_num 1, 2, or 4 depending on your base shape, or enter nothing for pgm g-code file')
    
    ###New from MULTIMATERIALV2 --------------------------------------------------------------------------------
    # for Printing on Aerotech:
  
    tool_axis = (["Z","C","B","A"] ) # don't change these
    line_pressures = [85,33,87,88]   # change these as you need; these are pressures for A,B,C and D
    com_ports = [1, 4, 1, 9]         # don't change these unless the COM ports changed; these are for the boxes on A, B, C, and D

    # Set current defaults around Z axis
    cur_tool_index = 0
    cur_tool = tool_axis[cur_tool_index]
    cur_pressure=line_pressures[cur_tool_index]
    cur_com_port = com_ports[cur_tool_index]

    def read_offset_file(self, file_name):
        global POS
        global s_POS
        """ 
        file = '.csv' contain offset of axis heads 
            from calibration_offset_v3.py

        return absolute positions for each z axis head, with the same height 
        using the default home of the printer as ref
        e.g POS is a Matrix contain x,y,z positions for all the
            Pos   = [[z x y], Z axis 
                    [z x y], C axis 
                    [z x y], B axis 
                    [z x y]] A axis 

        e.g s_POS is a Matrix contain x,y,z sensors readings for all the heads
            s_Pos   = [[x y z], Z axis 
                    [x y z], C axis 
                    [x y z], B axis 
                    [x y z]] A axis 
        """
        with open('{}'.format(file_name), 'rb') as f:
            reader = csv.reader(f)
            data = list(reader)
            
        ## for z ##
        zzz = data[1]
        zz = zzz[1:]

        z = [] 

        for item in zz:
            z.append(float(item))

        ## for c ##

        ccc = data[2]
        cc = ccc[1:]

        c = [] 

        for item in cc:
            c.append(float(item))

        ## for b ##

        bbb = data[3]
        bb = bbb[1:]

        b = [] 

        for item in bb:
            b.append(float(item))

        ## for a ##

        aaa = data[4]
        aa = aaa[1:]

        a = [] 

        for item in aa:
            a.append(float(item))

        ### Creating matrix from list ###

        ## Position Matrix 
        POS = [z[0:3],
        c[0:3],
        b[0:3],
        a[0:3]]

        ## Sensors Readings Matrix 
        s_POS = [z[3:],
        c[3:],
        b[3:],
        a[3:]]
        return POS,s_POS

    '''
    def set_cur_tool(self):
        """"Helper funciton. Sets the current axis letter, pressure, and com port to those corresponding the the current tool index."""
        # for Printing on Aerotech:
        tool_axis = (["Z","C","B","A"] ) # don't change these
        line_pressures = [85,33,87,88]   # change these as you need; these are pressures for A,B,C and D
        com_ports = [1, 4, 1, 9]         # don't change these unless the COM ports changed; these are for the boxes on A, B, C, and D

        # Set current defaults around Z axis
        cur_tool_index = 0
        cur_tool = tool_axis[cur_tool_index]
        cur_pressure=line_pressures[cur_tool_index]
        cur_com_port = com_ports[cur_tool_index]
        global cur_tool
        global cur_pressure
        global cur_com_port
        cur_tool = tool_axis[cur_tool_index]
        cur_com_port = com_ports[cur_tool_index]  
    '''

    def change_tool(self,to_tool_index):
        global POS
        """Change the current nozzle and default z axis."""
        self.write('; Changing tools...')
            
        
        if    (cur_tool_index != to_tool_index):
            global cur_tool_index
            global cur_tool
            global old_tool

            old_tool_index = cur_tool_index
            self.write(";cur tool " + str(cur_tool_index))
            self.write(";cur tool  " + str(cur_tool))
            self.write(";cur tool axis  " + str(tool_axis[cur_tool_index]))

            self.write(";cur tool " + str(cur_tool_index))
            self.write(";cur tool  " + str(cur_tool))
            self.write("; Change Tools from " + tool_axis[old_tool_index] + " to " + tool_axis[to_tool_index] + ".")

            self.rename_axis(z=tool_axis[to_tool_index])
            print("Renaming Z axis to " + cur_tool + ".")
            str_new_nozzle_offset_x = str(POS[to_tool_index][1]-POS[old_tool_index][1])
            str_new_nozzle_offset_y = str(POS[to_tool_index][2]-POS[old_tool_index][2])
            self.absolute()
            print ('New X -> ' + str_new_nozzle_offset_x + 'and new y -> '+ str_new_nozzle_offset_y)
            
            self.relative()
            self.write("G1 X" + str_new_nozzle_offset_x + " Y" + str_new_nozzle_offset_y + " ; set a new fixture offset to compensate for the new tools offset.")
            cur_tool_index = to_tool_index
            set_cur_tool()
        
        elif to_tool_index == cur_tool_index:
            print("ERROR: Switched to tool currently in use!")
            

        else: 
            raise RuntimeError ("tool index to use is out of range, please verify ")

    def set_tool_home(self,):

        self.abs_move(Z = 0 ,C = 0, B = 0, A = 0)

    def set_tool_offset(self,nz):
        """ If 2 - materials, use the z and c heads
            If 4 - materials does not matter """
        if nz == 2:
            self.abs_move(Z = POS[0][0] ,C = POS[1][0] , B = 0 , A = 0)
        elif nz == 3:
            self.abs_move(Z = POS[0][0] ,C = POS[1][0],B = POS[2][0],A = 0)
        elif nz  == 4: 
            self.abs_move(Z = POS[0][0] ,C = POS[1][0],B = POS[2][0],A = POS[3][0])
        else: 
            raise RuntimeError ('error setting the tool home ...')

    def remove_tool_offset(self):
        self.write('; Remove the tool offset in Z...')
        self.write("POSOFFSET CLEAR X Y U A B C Z")
        self.write("G53 ; clear any current fixture offset")
####---------------------------------------------------------------------------------------------------------------------------

####--ME500--Beibit-Blake---Spring'24------------------------------------------------------------------------------------------
    def calc_angle(self, coord_x, coord_y, centroid_x, centroid_y):
        '''Input x,y cartesian coords for a point and centroid. Calculates angle in degrees from x-axis '''
        angle = np.arctan2(coord_y-centroid_y, coord_x - centroid_x)
        return np.round(angle*(180/np.pi),9)
    
    def points_on_line(self,start,end,inc_d):
        '''
        Input start and end as tuple (x,y), inc_d as distance between points
        returns np.array of points along line connecting start and end, separated by distance inc_d
        '''
        x1, y1 = start[0],start[1]
        x2, y2 = end[0],end[1]
        
        #calculate total distance of line
        line_d = np.round(np.sqrt((x2-x1)**2+(y2-y1)**2),9)
        
        points = []
        d_count = 0
        if x1 == x2: #vertical line handling
            y = y1
            if y1 <= y2: 
                while y < y2:
                    points.append((x1,y))
                    y += inc_d
            else:    
                while y >= y2:
                    points.append((x1,y))
                    y -= inc_d
        else: 
            #Calculate amount to increment x and y
            unit_dx = (x2-x1)/line_d
            unit_dy = (y2-y1)/line_d
            #increment from x1 to x2 
            while d_count <= line_d:
                x = x1 + d_count*unit_dx
                y = y1 + d_count*unit_dy
                points.append((x,y))
                d_count += inc_d
        return np.array(points)
    
    def find_intersection(self, point1, point2, x0, y0):
        '''
        Input point1 and point2 as (x,y) tuples, x0, y0 as starting point
        From starting point (x0,y0), follows a line having slope m (based on print_angle)  
        Checks for an intersection in the line segment between point1 and point2
        returns None if no intersection is found
        returns intersection point as np.array if successful
        '''
        x1,y1 = point1 #line segment points
        x2,y2 = point2
        m = np.tan(np.radians(self.print_angle)) #slope from print_angle
        
        #Checks for parallel lines, hopefully avoiding inf and float errors
        #will probably come up somewhere else though
        dx1, dy1 = x2-x1, y2-y1 #direction vector for line segment
        dx2, dy2 = 1, m #direction vector for print line
        cross = dx1*dy2 - dx2*dy1 
        if cross == 0: 
            return None
        
        #calculate slope of line segment
        try:
            m1 = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            m1 = None

        if x1 == x2:  # Vertical segment
            x_int = x1
            y_int = m * (x_int - x0) + y0    
        elif self.print_angle == 90 or self.print_angle == 270: #vertical print
            x_int = x0
            y_int = m1*(x_int-x1)+y1
        elif y1 == y2:  # Horizontal segment
            y_int = y1
            x_int = (y1 - y0)/m + x0
        else:  # General case
            x_int = ((y1-y0)+m*x0-m1*x1)/(m-m1)
            y_int = m*(x_int-x0)+y0

        #Check intersection is within segment
        if min(x1,x2) <= x_int <= max(x1,x2):
            if min(y1,y2) <= y_int <= max(y1,y2):
                return np.array([x_int,y_int])
        else:
            return None

    def f1(self):
        #L to R horiztonal for rectangles
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        permimeter_coordinates = []

        #Flatten array st each row represents a vertex
        for vertex in my_mesh.vectors.reshape(-1,3): 
                    permimeter_coordinates.append((vertex[0],vertex[1],vertex[2]))

        #Delete z coords (1 layer only), sort, then take unique values   
        perim = np.array(permimeter_coordinates)
        perim = np.delete(perim, 2, 1)
        uniq_perim = []
        #np.unique() not working, loop through to find unique points
        for row in perim:
            current_tuple = tuple(row)
            if current_tuple not in uniq_perim:
                uniq_perim.append(current_tuple)

        #Sort vertices by angle around centroid of polygon
        centroid = [0,0]
        x_bin=0
        y_bin=0
        for i in range(len(uniq_perim)):
            x_bin += uniq_perim[i][0]
            y_bin += uniq_perim[i][1]
        centroid[0] = x_bin/len(uniq_perim)
        centroid[1] = y_bin/len(uniq_perim)
        #calculate angle from x axis for each vertex 
        angles = [self.calc_angle(coord[0],coord[1],centroid[0],centroid[1]) for coord in uniq_perim]
        i_sorted = np.argsort(angles)[::1] #sort CCW
        sorted_perim = np.array([np.round(uniq_perim[i],9) for i in i_sorted])
        #Roll perim array to start in correct corner
        #find index of bottom left most coordinate
        #Find indicies of min y coords 
        min_y_i = np.where(sorted_perim[:,1]==np.min(sorted_perim[:,1]))[0]
        i_sorted = np.argsort(sorted_perim[min_y_i,0])
        bottom_left_i = min_y_i[i_sorted[0]]
        sorted_perim = np.roll(sorted_perim,-bottom_left_i, axis=0)
        
        inc_y = self.nozzle_size*self.spacing 
        x_max = sorted_perim[1][0]
        y_max = sorted_perim[2][1]

        curr_x,curr_y = sorted_perim[0]
        self.set_home(0,0)
        while curr_y <= y_max:
            self.rel_move(z=5*self.nozzle_size)
            self.abs_move(curr_x+inc_y,curr_y)

            self.rel_move(z=-5*self.nozzle_size)
            self.pressure_on_pso('X')
            self.dwell(self.tiz)
            self.abs_move(x_max,curr_y)

            self.pressure_off_pso('X')
            self.dwell(self.tfz)
            curr_y += inc_y

        self.rel_move(z=5*self.nozzle_size)
        self.home()

    def f2(self):
        #R to L horiztonal for rectangles
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        permimeter_coordinates = []

        #Flatten array st each row represents a vertex
        for vertex in my_mesh.vectors.reshape(-1,3): 
                    permimeter_coordinates.append((vertex[0],vertex[1],vertex[2]))

        #Delete z coords (1 layer only), sort, then take unique values   
        perim = np.array(permimeter_coordinates)
        perim = np.delete(perim, 2, 1)
        uniq_perim = []
        #np.unique() not working, loop through to find unique points
        for row in perim:
            current_tuple = tuple(row)
            if current_tuple not in uniq_perim:
                uniq_perim.append(current_tuple)

        #Sort vertices by angle around centroid of polygon
        centroid = [0,0]
        x_bin=0
        y_bin=0
        for i in range(len(uniq_perim)):
            x_bin += uniq_perim[i][0]
            y_bin += uniq_perim[i][1]
        centroid[0] = x_bin/len(uniq_perim)
        centroid[1] = y_bin/len(uniq_perim)
        #calculate angle from x axis for each vertex 
        angles = [self.calc_angle(coord[0],coord[1],centroid[0],centroid[1]) for coord in uniq_perim]
        i_sorted = np.argsort(angles)[::1] #sort CCW
        sorted_perim = np.array([np.round(uniq_perim[i],9) for i in i_sorted])
        #Roll perim array to start in correct corner
        #find index of bottom left most coordinate
        #Find indicies of min y coords 
        min_y_i = np.where(sorted_perim[:,1]==np.min(sorted_perim[:,1]))[0]
        i_sorted = np.argsort(sorted_perim[min_y_i,0])
        bottom_right_i = min_y_i[np.argmax(sorted_perim[min_y_i,0])]
        sorted_perim = np.roll(sorted_perim,-bottom_right_i, axis=0)
        inc_y = self.nozzle_size*self.spacing 
        x_min = sorted_perim[2][0]
        y_max = sorted_perim[1][1]

        curr_x,curr_y = sorted_perim[0]
        self.set_home(0,0)
        while curr_y <= y_max:
            self.rel_move(z=5*self.nozzle_size)
            self.abs_move(curr_x,curr_y)

            self.rel_move(z=-5*self.nozzle_size)
            self.pressure_on_pso('X')
            self.dwell(self.tiz)
            self.abs_move(x_min+inc_y,curr_y)

            self.pressure_off_pso('X')
            self.dwell(self.tfz)
            curr_y += inc_y
        self.rel_move(z=5*self.nozzle_size)
        self.home()

    def f3(self):
        '''
        Print 1 layer in a single direction 
        self.print_angle defines print direction
        Does not work for concave shapes 
        Written for use with solenoid
        '''
        my_mesh = mesh.Mesh.from_file(str(self.stl_file)) #load in the STL file into a mesh
        permimeter_coordinates = []

        #Flatten array st each row represents a vertex
        for vertex in my_mesh.vectors.reshape(-1,3): 
                    permimeter_coordinates.append((vertex[0],vertex[1],vertex[2]))

        #Delete z coords (1 layer only), sort, then take unique values   
        perim = np.array(permimeter_coordinates)
        perim = np.delete(perim, 2, 1)
        uniq_perim = []
        #np.unique() not working, loop through to find unique points
        for row in perim:
            current_tuple = tuple(row)
            if current_tuple not in uniq_perim:
                uniq_perim.append(current_tuple)
        
        '''
        ~~~Sorting Section~~~~~~~~~~~~~~~~~~~~~~~~~
        For self.print_angle in range:
        (0,90) start bottom right, loop CW
        (90,180) start bottom left, loop CCW
        (180,270) Start bottom right, loop CCW
        (270,360) start bottom left, loop CW
        ''' 
        #Sort vertices by angle around centroid of polygon
        centroid = [0,0]
        x_bin=0
        y_bin=0
        for i in range(len(uniq_perim)):
            x_bin += uniq_perim[i][0]
            y_bin += uniq_perim[i][1]
        centroid[0] = x_bin/len(uniq_perim)
        centroid[1] = y_bin/len(uniq_perim)
        #calculate angle from x axis for each vertex 
        angles = [self.calc_angle(coord[0],coord[1],centroid[0],centroid[1]) for coord in uniq_perim]
        #Sort CW or CCW based on self.print_angle
        if (self.print_angle > 90 and self.print_angle < 270):
            i_sorted = np.argsort(angles)[::1] #CCW sort
        else:
            i_sorted = np.argsort(angles)[::-1] #CW sort
        sorted_perim = np.array([np.round(uniq_perim[i],9) for i in i_sorted])

        #Roll perim array to start in correct corner
        if self.print_angle > 0 and self.print_angle <= 90 or self.print_angle >= 180 and self.print_angle <= 270:
            #find index of bottom right most coordinate
            #find indices of minumum y coords
            min_y_i = np.where(sorted_perim[:,1]==np.min(sorted_perim[:,1]))[0]
            #of coords having min y vals, find index of max x val
            bottom_right_i = min_y_i[np.argmax(sorted_perim[min_y_i,0])]
            sorted_perim = np.roll(sorted_perim,-bottom_right_i, axis=0)
        else:
            #find index of bottom left most coordinate
            #Find indicies of min y coords again
            min_y_i = np.where(sorted_perim[:,1]==np.min(sorted_perim[:,1]))[0]
            i_sorted = np.argsort(sorted_perim[min_y_i,0])
            bottom_left_i = min_y_i[i_sorted[0]]
            sorted_perim = np.roll(sorted_perim,-bottom_left_i, axis=0)
       
        #distance to move gantry for each successive print
        inc_d = self.nozzle_size*self.spacing 

        #Build array of points representing perimeter, separated by inc_d
        perim_points = []
        for i in range(len(sorted_perim)):
            x1 = sorted_perim[i][0]
            y1 = sorted_perim[i][1]
            #modulus allows for shape to be closed within the loop
            x2 = sorted_perim[(i+1)%len(sorted_perim)][0] 
            y2 = sorted_perim[(i+1)%len(sorted_perim)][1]
            points = self.points_on_line((x1,y1),(x2,y2),inc_d)
            for i in range(len(points)):
                perim_points.append(points[i])
        perim_points = np.array(perim_points)
        
        '''
        Print Path Calculation
        Find the furthest intersection points of perim for each point in perim_points
        path_points has format [[start_point],[end_point]]
        '''
        path_points = [] 
        for i in range(math.floor((perim_points)/2)):
            temp = []
            x0, y0 = perim_points[i]
            #loop through all line segments in perimeter
            for j in range(len(perim_points)):
                if i != j:
                    p1 = perim_points[j]
                    p2 = perim_points[(j+1)%len(perim_points)]
                    intersection = self.find_intersection(p1, p2, x0, y0)
                    if intersection is not None:
                        #ignore intersections that are the starting point
                        if intersection is not np.allclose(intersection,[x0,y0],atol=1e-9):
                            temp.append(intersection)
            #filter out None values that snuck through
            valid_ints = [p for p in temp if p is not None]
            if valid_ints:
                #cheeky lambda function to find furthest point
                furthest_point = max(valid_ints, key=lambda p: np.linalg.norm(p-np.array([x0,y0])))
                path_points.append((np.array([x0,y0]),furthest_point))
        ''' 
        delete_me = []
        for i in range(len(path_points)):
            for j in range(len(path_points),-1):
                if path_points[i][0] == path_points[j][1]:
                    delete_me.append(path_points[j])
        path_points = np.delete(path_points,delete_me)
        print(np.shape(path_points)) '''
        '''BEGIN PRINT'''
        self.set_home(0,0)  
        seconds = 0
        togg_counter = 0
        for i in range(len(path_points)):
            start_x,start_y = path_points[i][0][0],path_points[i][0][1]
            end_x,end_y = path_points[i][1][0],path_points[i][1][1]
            #Lift nozzle and move to next start point
            self.rel_move(z=5*self.nozzle_size)
            self.abs_move(start_x,start_y)
            #Drop nozzle and begin print
            self.rel_move(z=-5*self.nozzle_size)

            self.pressure_on_pso('X')
            self.dwell(self.tiz) 
            self.abs_move(end_x,end_y)
            
            self.pressure_off_pso('X')
            self.dwell(self.tfz)

        self.home()

######--END-ME500----------------------------------------------------------------------------------------------------