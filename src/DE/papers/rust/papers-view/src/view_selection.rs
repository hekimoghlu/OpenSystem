use crate::ViewSelection;
use glib::translate::UnsafeFrom;

impl ViewSelection {
    #[inline]
    pub fn page(&self) -> i32 {
        unsafe { (*self.as_ptr()).page }
    }

    #[inline]
    pub fn rect(&self) -> papers_document::Rectangle {
        unsafe { papers_document::Rectangle::unsafe_from((*self.as_ptr()).rect) }
    }
}
