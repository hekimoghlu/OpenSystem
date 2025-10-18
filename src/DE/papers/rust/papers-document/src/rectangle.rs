use crate::Rectangle;
use std::fmt;

impl Rectangle {
    #[inline]
    pub fn x1(&self) -> f64 {
        self.inner.x1
    }

    #[inline]
    pub fn x2(&self) -> f64 {
        self.inner.x2
    }

    #[inline]
    pub fn y1(&self) -> f64 {
        self.inner.y1
    }

    #[inline]
    pub fn y2(&self) -> f64 {
        self.inner.y2
    }
}

impl fmt::Debug for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Rectangle")
            .field("x1", &self.x1())
            .field("x2", &self.x2())
            .field("y1", &self.y1())
            .field("y2", &self.y2())
            .finish()
    }
}
