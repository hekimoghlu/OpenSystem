Title: Integrating with cloud providers

Integration with cloud providers can be achieved by fetching a list of exported
cloud provider accounts using cloud_providers_collector_get_providers(). This function returns
a list of #CloudProvidersProvider objects that can then be used to obtain details about
the providers. Each #CloudProvidersProvider holds a list of #CloudProvidersAccount
that can be query using cloud_providers_provider_get_accounts().

To get notified about changes in either the #CloudProvidersProvider or each of
their #CloudProvidersAccount you can connect to the "notify::" signal of
each of their properties. Any UI elements should be updated after receiving
this signal.

Besides the account details, #CloudProvidersAccount may also export a #GMenuModel and a #GActionGroup
to provide actions that are related with the account. Those can be obtained by calling
cloud_providers_account_get_menu_model() and cloud_providers_account_get_action_group().
