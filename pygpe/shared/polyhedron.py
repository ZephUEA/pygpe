import polytope as pc
import numpy as np
from scipy.spatial import ConvexHull

def closest_point_on_segment(point, a, b):
    """
    Find the closest point on line segment [a, b] to the given point
    
    Args:
        point: 3D point
        a, b: endpoints of line segment
    
    Returns:
        Closest point on segment [a, b]
    """
    ab = b - a
    ap = point - a
    
    # Project onto line and clamp to segment
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    
    return a + t * ab

def closest_point_on_polygon(point, vertices, normal, d):
    """
    Find closest point on a convex polygon to the given point
    
    Args:
        point: 3D point
        vertices: ordered vertices of convex polygon
        normal: plane normal vector (assumed normalized)
        d: plane offset (normal · x = d)
    
    Returns:
        Closest point on the polygon (interior or boundary)
    """
    # Project point onto the plane
    distance = np.dot(point, normal) - d
    projection = point - distance * normal
    
    # Check if projection is inside the polygon using edge tests
    n = len(vertices)
    inside = True
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        to_point = projection - vertices[i]
        cross = np.cross(edge, to_point)
        if np.dot(cross, normal) < -1e-10:
            inside = False
            break
    
    if inside:
        return projection
    
    # Otherwise, find closest point on boundary
    min_dist = float('inf')
    closest = None
    
    for i in range(n):
        # Check edges
        edge_point = closest_point_on_segment(projection, vertices[i], vertices[(i + 1) % n])
        dist = np.linalg.norm(projection - edge_point)
        if dist < min_dist:
            min_dist = dist
            closest = edge_point
        
        # Check vertices
        dist = np.linalg.norm(projection - vertices[i])
        if dist < min_dist:
            min_dist = dist
            closest = vertices[i]
    
    return closest

class PolyhedronProjector:
    """
    Precomputed polyhedron projector for fast repeated projections
    """
    
    def __init__(self, poly):
        """
        Initialize with a polytope and precompute face information
        
        Args:
            poly: polytope.Polytope object in H-representation
        """
        self.polyhedron = poly
        # Compute vertices once
        vertices = pc.extreme(poly)
        
        # Compute convex hull once
        hull = ConvexHull(vertices)
        
        # Precompute all face data
        self.faces = []
        for simplex in hull.simplices:
            face_vertices = vertices[simplex]
            
            # Compute and store face normal and offset
            v1 = face_vertices[1] - face_vertices[0]
            v2 = face_vertices[2] - face_vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            d = np.dot(normal, face_vertices[0])
            
            self.faces.append({
                'vertices': face_vertices,
                'normal': normal,
                'd': d
            })
    
    def project(self, point):
        """
        Project a point onto the polyhedron
        
        Args:
            point: 3D point to project (assumed outside polyhedron)
        
        Returns:
            Closest point on the polyhedron surface
        """
        point = np.array(point)
        
        min_dist = float('inf')
        closest_point = None
        
        # Check each precomputed face
        for face in self.faces:
            face_closest = closest_point_on_polygon(
                point, 
                face['vertices'], 
                face['normal'], 
                face['d']
            )
            
            dist = np.linalg.norm(point - face_closest)
            if dist < min_dist:
                min_dist = dist
                closest_point = face_closest
        
        return closest_point