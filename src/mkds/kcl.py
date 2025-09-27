from .utils import (
    read_u16,
    read_vector_3d,
    read_u32,
    read_fx32
)


class KCL:
    """
    Represents a KCL (collision) file.

    KCL files store simplified model data for collision detection in games
    such as Mario Kart Wii / DS. They consist of a header, positions, normals,
    triangular prisms, and octree blocks.

    Header Layout
    -------------
    Offset  | Type  | Name                   | Description
    --------|-------|-----------------------|--------------------------------------------
    0x00    | u32   | positions_offset       | Offset to vertex positions section
    0x04    | u32   | normals_offset         | Offset to normal vectors section
    0x08    | u32   | prisms_offset          | Offset to triangular prisms section
    0x0C    | u32   | block_data_offset      | Offset to octree blocks
    0x10    | f32   | prism_thickness        | Depth of triangular prism along normal
    0x14    | Vec3  | area_min_pos           | Minimum coordinates of model bounding box
    0x20    | u32   | area_x_width_mask      | X-axis mask for octree
    0x24    | u32   | area_y_width_mask      | Y-axis mask for octree
    0x28    | u32   | area_z_width_mask      | Z-axis mask for octree
    0x2C    | u32   | block_width_shift      | Octree leaf block size shift
    0x30    | u32   | area_x_blocks_shift    | Root child index shift (Y axis)
    0x34    | u32   | area_xy_blocks_shift   | Root child index shift (Z axis)
    0x38    | f32?  | sphere_radius          | Optional: max sphere radius for collisions

    Attributes
    ----------
    _positions_offset : int
        File offset to position vectors
    _normals_offset : int
        File offset to normal vectors
    _prisms_offset : int
        File offset to prism data
    _block_data_offset : int
        File offset to octree blocks
    _prism_thickness : float
        Depth of each prism
    _area_min_pos : list[float]
        Minimum coordinates of the collision area
    _area_x_width_mask : int
        X-axis mask for octree
    _area_y_width_mask : int
        Y-axis mask for octree
    _area_z_width_mask : int
        Z-axis mask for octree
    _block_width_shift : int
        Octree leaf size shift
    _area_x_blocks_shift : int
        Root block child index shift (Y)
    _area_xy_blocks_shift : int
        Root block child index shift (Z)
    _sphere_radius : float or None
        Optional maximum sphere radius for collisions
    _prisms : Prisms
        Parsed prism objects
    _positions : list
        List of vertex positions
    _normals : list
        List of normal vectors
    """
    def __init__(self, data: bytes):
        self._data = data
        
        # Parsed Header
        self._positions_offset = read_u32(data, 0x00)
        self._normals_offset = read_u32(data, 0x04)
        self._prisms_offset = read_u32(data, 0x08)
        self._block_data_offset = read_u32(data, 0x0C)
        self._prism_thickness = read_fx32(data, 0x10)
        self._area_min_pos = read_vector_3d(data, 0x14)
        self._area_x_width_mask = read_u32(data, 0x20)
        self._area_y_width_mask = read_u32(data, 0x24)
        self._area_z_width_mask = read_u32(data, 0x28)
        self._block_width_shift = read_u32(data, 0x2C)
        self._area_x_blocks_shift = read_u32(data, 0x30)
        self._area_xy_blocks_shift = read_u32(data, 0x34)
        self._sphere_radius = None#read_f32(data, 0x38)
        
        # Parsed Data
        self._prisms = Prisms(data[self._prisms_offset+0x10:self._block_data_offset])
        self._positions = self._parse_positions(data)
        self._normals = self._parse_normals(data)

    def __str__(self):
        """
        Returns a human-readable summary of the KCL file.

        Output includes:
        - Number of positions and a preview of first and last vectors
        - Number of normals and a preview of first and last vectors
        - Number of prisms
        """
        def str_vec(l):
            return f"""
            {l[0]}
            ...
            {len(l)} vectors
            ...
            {l[-1]}
            """

        return f"""
        Positions:
        {str_vec(self._positions) if len(self._positions) != 0 else "No entries"}
        Normals:
        {str_vec(self._normals) if len(self._normals) != 0 else "No entries"}
        Prisms:
        {len(self._prisms) if len(self._prisms) != 0 else "No entries"}\n
        """

    @classmethod
    def from_file(cls, path, device=None):
        """
        Load a KCL file from disk.

        Parameters
        ----------
        path : str
            Path to the KCL file (default is DEFAULT_KCL_PATH)
        device : optional
            Ignored in this parser (kept for API consistency)

        Returns
        -------
        KCL
            Parsed KCL object
        """
        with open(path, 'rb') as f:
            data = f.read()
        return cls(data)

    def _parse_positions(self, data):
        """
        Parse position vectors from the file.

        Each position vector consists of 3 consecutive floats (X, Y, Z).
        The number of positions is determined from the prisms indices.

        Returns
        -------
        list
            List of 3D vectors (tuples of floats)
        """
        start = self._positions_offset
        section_size = max(self._prisms._pos_i)
        end = section_size * 0x0C + start + 0x0C
        d = data[start:end]
        return [read_vector_3d(d, i) for i in range(0, len(d), 0x0C)]

    def _parse_normals(self, data):
        """
        Parse normal vectors from the file.

        Each normal vector consists of 3 consecutive floats (X, Y, Z).
        The number of normals is determined from all prism normal indices.

        Returns
        -------
        list
            List of 3D normal vectors (tuples of floats)
        """
        start = self._normals_offset
        section_size = max([
            *self._prisms._fnrm_i,
            *self._prisms._enrm1_i,
            *self._prisms._enrm2_i,
            *self._prisms._enrm3_i,
        ])
        end = section_size * 0x0C + start + 0x0C
        d = data[start:end]
        return [read_vector_3d(d, i) for i in range(0, len(d), 0x0C)]


class Prisms:
    """
    Represents the triangular prisms section of the KCL file.

    Each prism is a 0x10 byte structure with the following layout:

    Offset | Type | Name      | Description
    -------|------|----------|---------------------------------------------
    0x00   | f32  | height   | Prism height from vertex 1 to opposite side
    0x04   | u16  | pos_i    | Index of first vertex in positions array
    0x06   | u16  | fnrm_i   | Face normal index
    0x08   | u16  | enrm1_i  | Edge normal A index
    0x0A   | u16  | enrm2_i  | Edge normal B index
    0x0C   | u16  | enrm3_i  | Edge normal C index
    0x0E   | u16  | attributes | Collision attributes

    Attributes
    ----------
    _height : list[float]
        Prism heights
    _pos_i : list[int]
        Vertex indices
    _fnrm_i : list[int]
        Face normal indices
    _enrm1_i : list[int]
        Edge normal 1 indices
    _enrm2_i : list[int]
        Edge normal 2 indices
    _enrm3_i : list[int]
        Edge normal 3 indices
    _attributes : list[int]
        Collision attribute flags
    """

    def __init__(self, data):
        self._iter = range(0, len(data), 0x10)
        self._height = [read_fx32(data[i:i + 0x10], 0x00) for i in self._iter]
        self._pos_i = [read_u16(data[i:i + 0x10], 0x04) for i in self._iter]
        self._fnrm_i = [read_u16(data[i:i + 0x10], 0x06) for i in self._iter]
        self._enrm1_i = [read_u16(data[i:i + 0x10], 0x08) for i in self._iter]
        self._enrm2_i = [read_u16(data[i:i + 0x10], 0x0A) for i in self._iter]
        self._enrm3_i = [read_u16(data[i:i + 0x10], 0x0C) for i in self._iter]
        self._attributes = [read_u16(data[i:i + 0x10], 0x0E) for i in self._iter]

    def __getitem__(self, idx):
        """
        Return all attributes of the prism at index `idx`.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
        return [arr[idx] for arr in self.__dict__.values()]

    def __len__(self):
        """
        Return the number of prisms in the section.
        """
        return len(self._pos_i)

    def __iter__(self):
        """
        Iterate over all prisms, yielding full attribute lists.
        """
        for i in self._iter:
            yield self[i // 0x10]

    
    

    
    