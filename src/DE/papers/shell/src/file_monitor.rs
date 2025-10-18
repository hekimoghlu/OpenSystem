use super::*;

mod imp {
    use super::*;

    #[derive(Properties, Default, Debug)]
    #[properties(wrapper_type = super::PpsFileMonitor)]
    pub struct PpsFileMonitor {
        #[property(construct_only, get)]
        pub(super) uri: RefCell<String>,
        pub(super) monitor: std::cell::OnceCell<gio::FileMonitor>,
        pub(super) timeout_id: RefCell<Option<glib::SourceId>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for PpsFileMonitor {
        const NAME: &'static str = "PpsFileMonitor";
        type Type = super::PpsFileMonitor;
        type ParentType = glib::Object;
    }

    #[glib::derived_properties]
    impl ObjectImpl for PpsFileMonitor {
        fn signals() -> &'static [Signal] {
            static SIGNALS: OnceLock<Vec<Signal>> = OnceLock::new();
            SIGNALS.get_or_init(|| vec![Signal::builder("changed").run_last().action().build()])
        }

        fn constructed(&self) {
            let file = gio::File::for_uri(self.uri.borrow().as_ref());
            match file.monitor_file(gio::FileMonitorFlags::NONE, gio::Cancellable::NONE) {
                Ok(monitor) => {
                    monitor.connect_changed(glib::clone!(
                        #[weak(rename_to = obj)]
                        self,
                        move |_, _, _, event| {
                            match event {
                                gio::FileMonitorEvent::ChangesDoneHint => {
                                    obj.timeout_stop();
                                    obj.obj().emit_by_name::<()>("changed", &[]);
                                }
                                gio::FileMonitorEvent::Changed => obj.timeout_start(),
                                _ => (),
                            }
                        }
                    ));

                    self.monitor.set(monitor).unwrap();
                }
                Err(e) => {
                    glib::g_warning!("", "{}", e.message());
                }
            }
        }

        fn dispose(&self) {
            self.timeout_stop();
        }
    }

    impl PpsFileMonitor {
        fn timeout_start(&self) {
            self.timeout_stop();

            let id = glib::timeout_add_seconds_local_once(
                5,
                glib::clone!(
                    #[weak(rename_to = obj)]
                    self,
                    move || {
                        obj.timeout_id.take();
                        obj.obj().emit_by_name::<()>("changed", &[]);
                    }
                ),
            );

            self.timeout_id.replace(Some(id));
        }

        fn timeout_stop(&self) {
            if let Some(id) = self.timeout_id.take() {
                id.remove();
            }
        }
    }
}

glib::wrapper! {
    pub struct PpsFileMonitor(ObjectSubclass<imp::PpsFileMonitor>);
}

impl PpsFileMonitor {
    pub fn new(uri: &str) -> Self {
        glib::Object::builder().property("uri", uri).build()
    }
}
