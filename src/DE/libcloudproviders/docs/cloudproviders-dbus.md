Title: DBus Interfaces

The DBus object that is passed to cloud_providers_provider_exporter_new() will
implement the org.freedesktop.DBus.ObjectManager interface, and that can be used
by clients to discover and query objects like the provider properties itself or
the accounts associated with your provider.
