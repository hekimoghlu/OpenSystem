use crate::deps::*;

mod imp {
    use super::*;

    #[derive(Properties, Default)]
    #[properties(wrapper_type = super::PpsThumbnailItem)]
    pub struct PpsThumbnailItem {
        #[property(get, set, nullable)]
        text: RefCell<String>,
        #[property(get, set, nullable)]
        paintable: RefCell<Option<gdk::Paintable>>,
        #[property(get, set, nullable)]
        binding: RefCell<Option<glib::Binding>>,
        #[property(get, set, nullable)]
        job: RefCell<Option<JobThumbnailTexture>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for PpsThumbnailItem {
        const NAME: &'static str = "PpsThumbnailItem";
        type Type = super::PpsThumbnailItem;
        type ParentType = glib::Object;
    }

    #[glib::derived_properties]
    impl ObjectImpl for PpsThumbnailItem {}
}

glib::wrapper! {
    pub struct PpsThumbnailItem(ObjectSubclass<imp::PpsThumbnailItem>);
}

impl PpsThumbnailItem {
    pub fn new() -> Self {
        glib::Object::builder().build()
    }
}

impl Default for PpsThumbnailItem {
    fn default() -> Self {
        Self::new()
    }
}
