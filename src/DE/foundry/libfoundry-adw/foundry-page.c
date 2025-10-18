/* foundry-page.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-page-private.h"
#include "foundry-workspace-private.h"

typedef struct
{
  GtkWidget          *content;
  GtkWidget          *auxillary;
  FoundryActionMuxer *muxer;
  GWeakRef            workspace_wr;
} FoundryPagePrivate;

typedef struct
{
  const FoundryAction *actions;
} FoundryPageClassPrivate;

enum {
  PROP_0,
  PROP_AUXILLARY,
  PROP_CAN_SAVE,
  PROP_CONTENT,
  PROP_ICON,
  PROP_SUBTITLE,
  PROP_TITLE,
  N_PROPS
};

enum {
  RAISE,
  N_SIGNALS
};

static void foundry_page_class_init (FoundryPageClass *klass);
static void foundry_page_init       (GTypeInstance    *instance,
                                     gpointer          g_class);

static GParamSpec *properties[N_PROPS];
static guint       signals[N_SIGNALS];
static int         FoundryPage_private_offset;
static gpointer    foundry_page_parent_class;

static inline gpointer
foundry_page_get_instance_private (FoundryPage *self)
{
  return (G_STRUCT_MEMBER_P (self, FoundryPage_private_offset));
}

static inline gpointer
foundry_page_class_get_private (FoundryPageClass *widget_class)
{
  return G_TYPE_CLASS_GET_PRIVATE (widget_class, FOUNDRY_TYPE_PAGE, FoundryPageClassPrivate);
}

GType
foundry_page_get_type (void)
{
  static GType widget_type = 0;

  if G_UNLIKELY (widget_type == 0)
    {
      const GTypeInfo widget_info =
      {
        sizeof (FoundryPageClass),
        NULL,
        NULL,
        (GClassInitFunc)foundry_page_class_init,
        NULL,
        NULL,
        sizeof (FoundryPage),
        0,
        foundry_page_init,
        NULL,
      };

      widget_type = g_type_register_static (GTK_TYPE_WIDGET,
                                            g_intern_static_string ("FoundryPage"),
                                            &widget_info,
                                            0);
      g_type_add_class_private (widget_type,
                                sizeof (FoundryPageClassPrivate));
      FoundryPage_private_offset = g_type_add_instance_private (widget_type,
                                                                sizeof (FoundryPagePrivate));
    }

  return widget_type;
}

static void
foundry_page_update_actions (FoundryPage *self)
{
  g_assert (FOUNDRY_IS_PAGE (self));

  foundry_page_action_set_enabled (self, "save", foundry_page_can_save (self));
}

static void
foundry_page_save_action (GtkWidget  *widget,
                          const char *action_name,
                          GVariant   *param)
{
  g_assert (FOUNDRY_IS_PAGE (widget));

  dex_future_disown (foundry_page_save (FOUNDRY_PAGE (widget)));
}

static void
foundry_page_focus_enter_cb (FoundryPage             *self,
                             GtkEventControllerFocus *controller)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  g_autoptr(FoundryWorkspace) workspace = NULL;

  g_assert (FOUNDRY_IS_PAGE (self));
  g_assert (GTK_IS_EVENT_CONTROLLER_FOCUS (controller));

  if ((workspace = g_weak_ref_get (&priv->workspace_wr)))
    _foundry_workspace_set_active_page (workspace, self);
}

static void
foundry_page_root (GtkWidget *widget)
{
  FoundryPage *self = (FoundryPage *)widget;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  GtkWidget *workspace;

  g_assert (FOUNDRY_IS_PAGE (self));

  GTK_WIDGET_CLASS (foundry_page_parent_class)->root (widget);

  if ((workspace = gtk_widget_get_ancestor (widget, FOUNDRY_TYPE_WORKSPACE)))
    g_weak_ref_set (&priv->workspace_wr, workspace);
}

static void
foundry_page_unroot (GtkWidget *widget)
{
  FoundryPage *self = (FoundryPage *)widget;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  g_autoptr(FoundryWorkspace) workspace = NULL;

  g_assert (FOUNDRY_IS_PAGE (self));

  if ((workspace = g_weak_ref_get (&priv->workspace_wr)))
    {
      if (foundry_workspace_get_active_page (workspace) == self)
        _foundry_workspace_set_active_page (workspace, NULL);

      g_weak_ref_set (&priv->workspace_wr, NULL);
    }

  GTK_WIDGET_CLASS (foundry_page_parent_class)->unroot (widget);
}

static void
foundry_page_measure (GtkWidget      *widget,
                      GtkOrientation  orientation,
                      int             for_size,
                      int            *minimum,
                      int            *natural,
                      int            *minimum_baseline,
                      int            *natural_baseline)
{
  FoundryPage *self = (FoundryPage *)widget;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_assert (FOUNDRY_IS_PAGE (self));

  if (priv->content != NULL)
    gtk_widget_measure (priv->content, orientation, for_size, minimum, natural, minimum_baseline, natural_baseline);
}

static void
foundry_page_size_allocate (GtkWidget *widget,
                            int        width,
                            int        height,
                            int        baseline)
{
  FoundryPage *self = (FoundryPage *)widget;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_assert (FOUNDRY_IS_PAGE (self));

  if (priv->content != NULL)
    gtk_widget_size_allocate (priv->content,
                              &(GtkAllocation) { 0, 0, width, height },
                              baseline);
}

static void
foundry_page_constructed (GObject *object)
{
  FoundryPage *self = (FoundryPage *)object;
  FoundryPageClass *page_class = FOUNDRY_PAGE_GET_CLASS (self);
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  FoundryPageClassPrivate *class_priv = foundry_page_class_get_private (page_class);

  G_OBJECT_CLASS (foundry_page_parent_class)->constructed (object);

  foundry_action_muxer_connect_actions (priv->muxer, self, class_priv->actions);

  gtk_widget_insert_action_group (GTK_WIDGET (self), "page", G_ACTION_GROUP (priv->muxer));
}

static void
foundry_page_dispose (GObject *object)
{
  FoundryPage *self = (FoundryPage *)object;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  GtkWidget *child;

  foundry_action_muxer_remove_all (priv->muxer);

  g_clear_object (&priv->auxillary);

  priv->content = NULL;

  while ((child = gtk_widget_get_first_child (GTK_WIDGET (self))))
    gtk_widget_unparent (child);

  g_weak_ref_set (&priv->workspace_wr, NULL);

  G_OBJECT_CLASS (foundry_page_parent_class)->dispose (object);
}

static void
foundry_page_finalize (GObject *object)
{
  FoundryPage *self = (FoundryPage *)object;
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_clear_object (&priv->muxer);
  g_weak_ref_clear (&priv->workspace_wr);

  G_OBJECT_CLASS (foundry_page_parent_class)->finalize (object);
}

static void
foundry_page_get_property (GObject    *object,
                           guint       prop_id,
                           GValue     *value,
                           GParamSpec *pspec)
{
  FoundryPage *self = FOUNDRY_PAGE (object);

  switch (prop_id)
    {
    case PROP_AUXILLARY:
      g_value_set_object (value, foundry_page_get_auxillary (self));
      break;

    case PROP_CAN_SAVE:
      g_value_set_boolean (value, foundry_page_can_save (self));
      break;

    case PROP_CONTENT:
      g_value_set_object (value, foundry_page_get_content (self));
      break;

    case PROP_ICON:
      g_value_take_object (value, foundry_page_dup_icon (self));
      break;

    case PROP_SUBTITLE:
      g_value_take_string (value, foundry_page_dup_subtitle (self));
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_page_dup_title (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_page_set_property (GObject      *object,
                           guint         prop_id,
                           const GValue *value,
                           GParamSpec   *pspec)
{
  FoundryPage *self = FOUNDRY_PAGE (object);

  switch (prop_id)
    {
    case PROP_AUXILLARY:
      foundry_page_set_auxillary (self, g_value_get_object (value));
      break;

    case PROP_CONTENT:
      foundry_page_set_content (self, g_value_get_object (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_page_class_init (FoundryPageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GtkWidgetClass *widget_class = GTK_WIDGET_CLASS (klass);

  g_type_class_adjust_private_offset (klass, &FoundryPage_private_offset);
  foundry_page_parent_class = g_type_class_peek_parent (klass);

  object_class->constructed = foundry_page_constructed;
  object_class->dispose = foundry_page_dispose;
  object_class->finalize = foundry_page_finalize;
  object_class->get_property = foundry_page_get_property;
  object_class->set_property = foundry_page_set_property;

  widget_class->measure = foundry_page_measure;
  widget_class->size_allocate = foundry_page_size_allocate;
  widget_class->root = foundry_page_root;
  widget_class->unroot = foundry_page_unroot;

  signals[RAISE] =
    g_signal_new ("raise",
                  G_TYPE_FROM_CLASS (klass),
                  G_SIGNAL_RUN_LAST,
                  0,
                  NULL, NULL,
                  NULL,
                  G_TYPE_NONE, 0);

  properties[PROP_AUXILLARY] =
    g_param_spec_object ("auxillary", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_CAN_SAVE] =
    g_param_spec_boolean ("can-save", NULL, NULL,
                          FALSE,
                          (G_PARAM_READABLE |
                           G_PARAM_STATIC_STRINGS));

  properties[PROP_CONTENT] =
    g_param_spec_object ("content", NULL, NULL,
                         GTK_TYPE_WIDGET,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ICON] =
    g_param_spec_object ("icon", NULL, NULL,
                         G_TYPE_ICON,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SUBTITLE] =
    g_param_spec_string ("subtitle", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);

  gtk_widget_class_set_css_name (widget_class, "page");

  foundry_page_class_install_action (klass, "save", NULL, foundry_page_save_action);
}

static void
foundry_page_init (GTypeInstance *instance,
                   gpointer       g_class)
{
  FoundryPage *self = FOUNDRY_PAGE (instance);
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  GtkEventController *controller;

  g_weak_ref_init (&priv->workspace_wr, NULL);

  priv->muxer = foundry_action_muxer_new ();

  g_signal_connect (self,
                    "notify::can-save",
                    G_CALLBACK (foundry_page_update_actions),
                    NULL);

  controller = gtk_event_controller_focus_new ();
  g_signal_connect_object (controller,
                           "enter",
                           G_CALLBACK (foundry_page_focus_enter_cb),
                           self,
                           G_CONNECT_SWAPPED);
  gtk_widget_add_controller (GTK_WIDGET (self), g_steal_pointer (&controller));

  foundry_page_update_actions (self);
}

static void
foundry_page_class_add_action (FoundryPageClass *page_class,
                               FoundryAction    *action)
{
  FoundryPageClassPrivate *class_priv = foundry_page_class_get_private (page_class);

  g_assert (FOUNDRY_IS_PAGE_CLASS (page_class));
  g_assert (action != NULL);
  g_assert (action->next == NULL);
  g_assert (action->position == 0);

  /* Precalculate action "position". To be stable this is the
   * number of items from the end.
   */
  for (const FoundryAction *iter = class_priv->actions;
       iter != NULL;
       iter = iter->next)
    action->position++;

  action->next = class_priv->actions;
  class_priv->actions = action;
}

/**
 * foundry_page_class_install_action:
 * @activate: (scope forever):
 *
 */
void
foundry_page_class_install_action (FoundryPageClass            *page_class,
                                   const char                  *action_name,
                                   const char                  *parameter_type,
                                   GtkWidgetActionActivateFunc  activate)
{
  FoundryAction *action;

  g_return_if_fail (FOUNDRY_IS_PAGE_CLASS (page_class));
  g_return_if_fail (action_name != NULL);
  g_return_if_fail (activate != NULL);

  action = g_new0 (FoundryAction, 1);
  action->owner = G_TYPE_FROM_CLASS (page_class);
  action->name = g_intern_string (action_name);
  if (parameter_type != NULL)
    action->parameter_type = g_variant_type_new (parameter_type);
  action->activate = (FoundryActionActivateFunc)activate;

  foundry_page_class_add_action (page_class, action);
}

static const GVariantType *
determine_type (GParamSpec *pspec)
{
  if (G_TYPE_IS_ENUM (pspec->value_type))
    return G_VARIANT_TYPE_STRING;

  switch (pspec->value_type)
    {
    case G_TYPE_BOOLEAN:
      return G_VARIANT_TYPE_BOOLEAN;

    case G_TYPE_INT:
      return G_VARIANT_TYPE_INT32;

    case G_TYPE_UINT:
      return G_VARIANT_TYPE_UINT32;

    case G_TYPE_DOUBLE:
    case G_TYPE_FLOAT:
      return G_VARIANT_TYPE_DOUBLE;

    case G_TYPE_STRING:
      return G_VARIANT_TYPE_STRING;

    default:
      g_critical ("Unable to use foundry_page_class_install_property_action with property '%s:%s' of type '%s'",
                  g_type_name (pspec->owner_type), pspec->name, g_type_name (pspec->value_type));
      return NULL;
    }
}

/**
 * foundry_page_class_install_property_action:
 * @page_class: a `FoundryPageClass`
 * @action_name: name of the action
 * @property_name: name of the property in instances of @page_class
 *   or any parent class.
 *
 * Installs an action called @action_name on @page_class and
 * binds its state to the value of the @property_name property.
 *
 * This function will perform a few santity checks on the property selected
 * via @property_name. Namely, the property must exist, must be readable,
 * writable and must not be construct-only. There are also restrictions
 * on the type of the given property, it must be boolean, int, unsigned int,
 * double or string. If any of these conditions are not met, a critical
 * warning will be printed and no action will be added.
 *
 * The state type of the action matches the property type.
 *
 * If the property is boolean, the action will have no parameter and
 * toggle the property value. Otherwise, the action will have a parameter
 * of the same type as the property.
 */
void
foundry_page_class_install_property_action (FoundryPageClass *page_class,
                                            const char       *action_name,
                                            const char       *property_name)
{
  const GVariantType *state_type;
  FoundryAction *action;
  GParamSpec *pspec;

  g_return_if_fail (GTK_IS_WIDGET_CLASS (page_class));

  if (!(pspec = g_object_class_find_property (G_OBJECT_CLASS (page_class), property_name)))
    {
      g_critical ("Attempted to use non-existent property '%s:%s' for foundry_page_class_install_property_action",
                  G_OBJECT_CLASS_NAME (page_class), property_name);
      return;
    }

  if (~pspec->flags & G_PARAM_READABLE || ~pspec->flags & G_PARAM_WRITABLE || pspec->flags & G_PARAM_CONSTRUCT_ONLY)
    {
      g_critical ("Property '%s:%s' used with foundry_page_class_install_property_action must be readable, writable, and not construct-only",
                  G_OBJECT_CLASS_NAME (page_class), property_name);
      return;
    }

  state_type = determine_type (pspec);

  if (!state_type)
    return;

  action = g_new0 (FoundryAction, 1);
  action->owner = G_TYPE_FROM_CLASS (page_class);
  action->name = g_intern_string (action_name);
  action->pspec = pspec;
  action->state_type = state_type;
  if (action->pspec->value_type != G_TYPE_BOOLEAN)
    action->parameter_type = action->state_type;

  foundry_page_class_add_action (page_class, action);
}

void
foundry_page_action_set_enabled (FoundryPage *self,
                                 const char  *action_name,
                                 gboolean     enabled)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);
  FoundryPageClassPrivate *class_priv;

  g_return_if_fail (FOUNDRY_IS_PAGE (self));
  g_return_if_fail (action_name != NULL);

  class_priv = foundry_page_class_get_private (FOUNDRY_PAGE_GET_CLASS (self));

  for (const FoundryAction *iter = class_priv->actions; iter; iter = iter->next)
    {
      if (g_strcmp0 (iter->name, action_name) == 0)
        {
          foundry_action_muxer_set_enabled (priv->muxer, iter, enabled);
          return;
        }
    }

  g_warning ("Failed to locate action `%s` in %s",
             action_name, G_OBJECT_TYPE_NAME (self));
}

/**
 * foundry_page_get_content:
 * @self: a [class@FoundryAdw.Page]
 *
 * Returns: (transfer none) (nullable): the content widget
 */
GtkWidget *
foundry_page_get_content (FoundryPage *self)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  return priv->content;
}

void
foundry_page_set_content (FoundryPage *self,
                          GtkWidget   *content)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_PAGE (self));
  g_return_if_fail (!content || GTK_IS_WIDGET (content));
  g_return_if_fail (!content || gtk_widget_get_parent (content) == NULL);

  if (content == priv->content)
    return;

  if (content != NULL)
    gtk_widget_set_parent (content, GTK_WIDGET (self));

  g_clear_pointer (&priv->content, gtk_widget_unparent);
  priv->content = content;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_CONTENT]);
}

/**
 * foundry_page_get_auxillary:
 * @self: a [class@FoundryAdw.Page]
 *
 * Gets the auxillary widget for the page, if any.
 *
 * Returns: (transfer none) (nullable):
 */
GtkWidget *
foundry_page_get_auxillary (FoundryPage *self)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  return priv->auxillary;
}

void
foundry_page_set_auxillary (FoundryPage *self,
                            GtkWidget   *auxillary)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_PAGE (self));
  g_return_if_fail (!auxillary || GTK_IS_WIDGET (auxillary));
  g_return_if_fail (!auxillary || gtk_widget_get_parent (auxillary) == NULL);

  if (auxillary == priv->auxillary)
    return;

  if (auxillary)
    g_object_ref_sink (auxillary);

  g_clear_object (&priv->auxillary);
  priv->auxillary = auxillary;

  g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_AUXILLARY]);
}

/**
 * foundry_page_dup_subtitle:
 * @self: a [class@FoundryAdw.Page]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_page_dup_subtitle (FoundryPage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  if (FOUNDRY_PAGE_GET_CLASS (self)->dup_subtitle)
    return FOUNDRY_PAGE_GET_CLASS (self)->dup_subtitle (self);

  return NULL;
}

/**
 * foundry_page_dup_title:
 * @self: a [class@FoundryAdw.Page]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_page_dup_title (FoundryPage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  if (FOUNDRY_PAGE_GET_CLASS (self)->dup_title)
    return FOUNDRY_PAGE_GET_CLASS (self)->dup_title (self);

  return NULL;
}

/**
 * foundry_page_dup_icon:
 * @self: a [class@FoundryAdw.Page]
 *
 * Returns: (transfer full) (nullable):
 */
GIcon *
foundry_page_dup_icon (FoundryPage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  if (FOUNDRY_PAGE_GET_CLASS (self)->dup_icon)
    return FOUNDRY_PAGE_GET_CLASS (self)->dup_icon (self);

  return NULL;
}

void
foundry_page_raise (FoundryPage *self)
{
  g_return_if_fail (FOUNDRY_IS_PAGE (self));

  g_signal_emit (self, signals[RAISE], 0);
}

/**
 * foundry_page_can_save:
 * @self: a [class@FoundryAdw.Page]
 *
 * Checks if the page can be saved.
 *
 * Implementations of [class@FoundryAdw.Page] should notify the `can-save`
 * property when the value of this function changes.
 *
 * Returns: %TRUE if the page can be saved
 */
gboolean
foundry_page_can_save (FoundryPage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), FALSE);

  if (FOUNDRY_PAGE_GET_CLASS (self)->can_save)
    return FOUNDRY_PAGE_GET_CLASS (self)->can_save (self);

  return FALSE;
}

/**
 * foundry_page_save:
 * @self: a [class@FoundryAdw.Page]
 *
 * Requests that the page saves.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value when the operation has completed or rejects with error.
 */
DexFuture *
foundry_page_save (FoundryPage *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_PAGE (self));

  if (FOUNDRY_PAGE_GET_CLASS (self)->save)
    return FOUNDRY_PAGE_GET_CLASS (self)->save (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

/**
 * foundry_page_save_as:
 * @self: a [class@FoundryAdw.Page]
 *
 * Requests that the page saves to a new file after querying the user
 * for the necessary new file.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to any
 *   value when the operation has completed or rejects with error.
 */
DexFuture *
foundry_page_save_as (FoundryPage *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_PAGE (self));

  if (FOUNDRY_PAGE_GET_CLASS (self)->save_as)
    return FOUNDRY_PAGE_GET_CLASS (self)->save_as (self);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_NOT_SUPPORTED,
                                "Not supported");
}

FoundryActionMuxer *
_foundry_page_get_action_muxer (FoundryPage *self)
{
  FoundryPagePrivate *priv = foundry_page_get_instance_private (self);

  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  return priv->muxer;
}

/**
 * foundry_page_dup_menu:
 * @self: a [class@FoundryAdw.Page]
 *
 * Returns: (transfer full) (nullable):
 */
GMenuModel *
foundry_page_dup_menu (FoundryPage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_PAGE (self), NULL);

  if (FOUNDRY_PAGE_GET_CLASS (self)->dup_menu)
    return FOUNDRY_PAGE_GET_CLASS (self)->dup_menu (self);

  return NULL;
}
