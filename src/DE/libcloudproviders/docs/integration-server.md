Title: Implementing support as a cloud provider

Cloud providers need to create #CloudProvidersAccountExporter objects for
every account they like to expose to the API and use the #CloudProvidersAccountExporter
API to set the properties of the accounts. Also a #CloudProvidersProviderExporter
needs to be created in order to hold all the #CloudProvidersAccountExporter objects
that will be added to the provider when created. #CloudProvidersProviderExporter also
exports properties that define the provider that can be set using the #CloudProvidersProviderExporter API.

To register your cloud provider, you need to expose the interface it is implementing in its desktop file

Example desktop file to register a cloud provider:
```ini
[Desktop Entry]
Type=Application
Name=mycloudprovider example server
NoDisplay=true
Implements=org.freedesktop.CloudProviders

[org.freedesktop.CloudProviders]
BusName=org.mycloudprovider.CloudProviders.ServerExample
ObjectPath=/org/mycloudprovider/CloudProviders/ServerExample
```

In previous versions, to register your cloud provider you needed to provide a file in DATADIR/cloud-providers.
This way of registering providers is still working but is not compatible with containerization.

Example file to register a cloud provider in previous versions:
```ini
[Cloud Provider]
BusName=org.mycloudprovider.CloudProviders.ServerExample
ObjectPath=/org/mycloudprovider/CloudProviders/ServerExample
Version=1
```
