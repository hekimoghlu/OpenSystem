/* foundry-flatpak-options.c
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

#include <foundry.h>

#include "foundry-flatpak-options.h"
#include "foundry-flatpak-arch-options.h"

struct _FoundryFlatpakOptions
{
  FoundryFlatpakSerializable   parent_instance;

  FoundryFlatpakArchOptions   *arch;

  char                        *append_ld_library_path;
  char                        *append_path;
  char                        *append_pkg_config_path;
  char                        *cflags;
  char                        *cppflags;
  char                        *cxxflags;
  char                        *ldflags;
  char                        *libdir;
  char                        *prefix;
  char                        *prepend_ld_library_path;
  char                        *prepend_path;
  char                        *prepend_pkg_config_path;

  char                       **build_args;
  char                       **config_opts;
  char                       **env;
  char                       **make_args;
  char                       **make_install_args;
  char                       **secret_env;
  char                       **secret_opts;
  char                       **test_args;

  guint                        cflags_override : 1;
  guint                        cppflags_override : 1;
  guint                        cxxflags_override : 1;
  guint                        ldflags_override : 1;
  guint                        no_debuginfo : 1;
  guint                        no_debuginfo_compression : 1;
  guint                        strip : 1;
};

enum {
  PROP_0,
  PROP_CFLAGS,
  PROP_CFLAGS_OVERRIDE,
  PROP_CPPFLAGS,
  PROP_CPPFLAGS_OVERRIDE,
  PROP_CXXFLAGS,
  PROP_CXXFLAGS_OVERRIDE,
  PROP_LDFLAGS,
  PROP_LDFLAGS_OVERRIDE,
  PROP_PREFIX,
  PROP_LIBDIR,
  PROP_ENV,
  PROP_STRIP,
  PROP_NO_DEBUGINFO,
  PROP_NO_DEBUGINFO_COMPRESSION,
  PROP_ARCH,
  PROP_BUILD_ARGS,
  PROP_TEST_ARGS,
  PROP_CONFIG_OPTS,
  PROP_SECRET_OPTS,
  PROP_SECRET_ENV,
  PROP_MAKE_ARGS,
  PROP_MAKE_INSTALL_ARGS,
  PROP_APPEND_PATH,
  PROP_PREPEND_PATH,
  PROP_APPEND_LD_LIBRARY_PATH,
  PROP_PREPEND_LD_LIBRARY_PATH,
  PROP_APPEND_PKG_CONFIG_PATH,
  PROP_PREPEND_PKG_CONFIG_PATH,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakOptions, foundry_flatpak_options, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

static DexFuture *
foundry_flatpak_options_deserialize_property (FoundryFlatpakSerializable *serializable,
                                              const char                 *property_name,
                                              JsonNode                   *property_node)
{
  FoundryFlatpakOptions *self = FOUNDRY_FLATPAK_OPTIONS (serializable);

  if (g_strcmp0 (property_name, "env") == 0 &&
      JSON_NODE_HOLDS_OBJECT (property_node))
    {
      JsonObject *object = json_node_get_object (property_node);
      const char *key;
      JsonObjectIter iter;
      JsonNode *value_node;
      g_auto(GStrv) environ = NULL;

      json_object_iter_init (&iter, object);
      while (json_object_iter_next (&iter, &key, &value_node))
        {
          if (JSON_NODE_HOLDS_VALUE (value_node) &&
              G_TYPE_STRING == json_node_get_value_type (value_node))
            environ = g_environ_setenv (environ, key, json_node_get_string (value_node), TRUE);
        }

      foundry_set_strv (&self->env, (const char * const *)environ);

      return dex_future_new_true ();
    }

  return FOUNDRY_FLATPAK_SERIALIZABLE_CLASS (foundry_flatpak_options_parent_class)->
    deserialize_property (serializable, property_name, property_node);
}

static void
foundry_flatpak_options_finalize (GObject *object)
{
  FoundryFlatpakOptions *self = (FoundryFlatpakOptions *)object;

  g_clear_pointer (&self->cflags, g_free);
  g_clear_pointer (&self->cxxflags, g_free);
  g_clear_pointer (&self->cppflags, g_free);
  g_clear_pointer (&self->ldflags, g_free);
  g_clear_pointer (&self->append_path, g_free);
  g_clear_pointer (&self->prepend_path, g_free);
  g_clear_pointer (&self->append_ld_library_path, g_free);
  g_clear_pointer (&self->prepend_ld_library_path, g_free);
  g_clear_pointer (&self->append_pkg_config_path, g_free);
  g_clear_pointer (&self->prepend_pkg_config_path, g_free);
  g_clear_pointer (&self->prefix, g_free);
  g_clear_pointer (&self->libdir, g_free);

  g_clear_pointer (&self->env, g_strfreev);
  g_clear_pointer (&self->build_args, g_strfreev);
  g_clear_pointer (&self->test_args, g_strfreev);
  g_clear_pointer (&self->config_opts, g_strfreev);
  g_clear_pointer (&self->secret_opts, g_strfreev);
  g_clear_pointer (&self->secret_env, g_strfreev);
  g_clear_pointer (&self->make_args, g_strfreev);
  g_clear_pointer (&self->make_install_args, g_strfreev);

  g_clear_object (&self->arch);

  G_OBJECT_CLASS (foundry_flatpak_options_parent_class)->finalize (object);
}

static void
foundry_flatpak_options_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryFlatpakOptions *self = FOUNDRY_FLATPAK_OPTIONS (object);

  switch (prop_id)
    {
    case PROP_CFLAGS:
      g_value_set_string (value, self->cflags);
      break;

    case PROP_CFLAGS_OVERRIDE:
      g_value_set_boolean (value, self->cflags_override);
      break;

    case PROP_CPPFLAGS:
      g_value_set_string (value, self->cppflags);
      break;

    case PROP_CPPFLAGS_OVERRIDE:
      g_value_set_boolean (value, self->cppflags_override);
      break;

    case PROP_CXXFLAGS:
      g_value_set_string (value, self->cxxflags);
      break;

    case PROP_CXXFLAGS_OVERRIDE:
      g_value_set_boolean (value, self->cxxflags_override);
      break;

    case PROP_LDFLAGS:
      g_value_set_string (value, self->ldflags);
      break;

    case PROP_LDFLAGS_OVERRIDE:
      g_value_set_boolean (value, self->ldflags_override);
      break;

    case PROP_APPEND_PATH:
      g_value_set_string (value, self->append_path);
      break;

    case PROP_PREPEND_PATH:
      g_value_set_string (value, self->prepend_path);
      break;

    case PROP_APPEND_LD_LIBRARY_PATH:
      g_value_set_string (value, self->append_ld_library_path);
      break;

    case PROP_PREPEND_LD_LIBRARY_PATH:
      g_value_set_string (value, self->prepend_ld_library_path);
      break;

    case PROP_APPEND_PKG_CONFIG_PATH:
      g_value_set_string (value, self->append_pkg_config_path);
      break;

    case PROP_PREPEND_PKG_CONFIG_PATH:
      g_value_set_string (value, self->prepend_pkg_config_path);
      break;

    case PROP_PREFIX:
      g_value_set_string (value, self->prefix);
      break;

    case PROP_LIBDIR:
      g_value_set_string (value, self->libdir);
      break;

    case PROP_ENV:
      g_value_set_boxed (value, self->env);
      break;

    case PROP_ARCH:
      g_value_set_object (value, self->arch);
      break;

    case PROP_BUILD_ARGS:
      g_value_set_boxed (value, self->build_args);
      break;

    case PROP_TEST_ARGS:
      g_value_set_boxed (value, self->test_args);
      break;

    case PROP_CONFIG_OPTS:
      g_value_set_boxed (value, self->config_opts);
      break;

    case PROP_SECRET_OPTS:
      g_value_set_boxed (value, self->secret_opts);
      break;

    case PROP_SECRET_ENV:
      g_value_set_boxed (value, self->secret_env);
      break;

    case PROP_MAKE_ARGS:
      g_value_set_boxed (value, self->make_args);
      break;

    case PROP_MAKE_INSTALL_ARGS:
      g_value_set_boxed (value, self->make_install_args);
      break;

    case PROP_STRIP:
      g_value_set_boolean (value, self->strip);
      break;

    case PROP_NO_DEBUGINFO:
      g_value_set_boolean (value, self->no_debuginfo);
      break;

    case PROP_NO_DEBUGINFO_COMPRESSION:
      g_value_set_boolean (value, self->no_debuginfo_compression);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_options_set_property (GObject      *object,
                                      guint         prop_id,
                                      const GValue *value,
                                      GParamSpec   *pspec)
{
  FoundryFlatpakOptions *self = FOUNDRY_FLATPAK_OPTIONS (object);

  switch (prop_id)
    {
    case PROP_CFLAGS:
      g_set_str (&self->cflags, g_value_get_string (value));
      break;

    case PROP_CFLAGS_OVERRIDE:
      self->cflags_override = g_value_get_boolean (value);
      break;

    case PROP_CXXFLAGS:
      g_set_str (&self->cxxflags, g_value_get_string (value));
      break;

    case PROP_CXXFLAGS_OVERRIDE:
      self->cxxflags_override = g_value_get_boolean (value);
      break;

    case PROP_CPPFLAGS:
      g_set_str (&self->cppflags, g_value_get_string (value));
      break;

    case PROP_CPPFLAGS_OVERRIDE:
      self->cppflags_override = g_value_get_boolean (value);
      break;

    case PROP_LDFLAGS:
      g_set_str (&self->ldflags, g_value_get_string (value));
      break;

    case PROP_LDFLAGS_OVERRIDE:
      self->ldflags_override = g_value_get_boolean (value);
      break;

    case PROP_APPEND_PATH:
      g_set_str (&self->append_path, g_value_get_string (value));
      break;

    case PROP_PREPEND_PATH:
      g_set_str (&self->prepend_path, g_value_get_string (value));
      break;

    case PROP_APPEND_LD_LIBRARY_PATH:
      g_set_str (&self->append_ld_library_path, g_value_get_string (value));
      break;

    case PROP_PREPEND_LD_LIBRARY_PATH:
      g_set_str (&self->prepend_ld_library_path, g_value_get_string (value));
      break;

    case PROP_APPEND_PKG_CONFIG_PATH:
      g_set_str (&self->append_pkg_config_path, g_value_get_string (value));
      break;

    case PROP_PREPEND_PKG_CONFIG_PATH:
      g_set_str (&self->prepend_pkg_config_path, g_value_get_string (value));
      break;

    case PROP_PREFIX:
      g_set_str (&self->prefix, g_value_get_string (value));
      break;

    case PROP_LIBDIR:
      g_set_str (&self->libdir, g_value_get_string (value));
      break;

    case PROP_ENV:
      foundry_set_strv (&self->env, g_value_get_boxed (value));
      break;

    case PROP_ARCH:
      g_set_object (&self->arch, g_value_get_object (value));
      break;

    case PROP_BUILD_ARGS:
      foundry_set_strv (&self->build_args, g_value_get_boxed (value));
      break;

    case PROP_TEST_ARGS:
      foundry_set_strv (&self->test_args, g_value_get_boxed (value));
      break;

    case PROP_CONFIG_OPTS:
      foundry_set_strv (&self->config_opts, g_value_get_boxed (value));
      break;

    case PROP_SECRET_OPTS:
      foundry_set_strv (&self->secret_opts, g_value_get_boxed (value));
      break;

    case PROP_SECRET_ENV:
      foundry_set_strv (&self->secret_env, g_value_get_boxed (value));
      break;

    case PROP_MAKE_ARGS:
      foundry_set_strv (&self->make_args, g_value_get_boxed (value));
      break;

    case PROP_MAKE_INSTALL_ARGS:
      foundry_set_strv (&self->make_install_args, g_value_get_boxed (value));
      break;

    case PROP_STRIP:
      self->strip = g_value_get_boolean (value);
      break;

    case PROP_NO_DEBUGINFO:
      self->no_debuginfo = g_value_get_boolean (value);
      break;

    case PROP_NO_DEBUGINFO_COMPRESSION:
      self->no_debuginfo_compression = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_options_class_init (FoundryFlatpakOptionsClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSerializableClass *serializable_class = FOUNDRY_FLATPAK_SERIALIZABLE_CLASS (klass);

  object_class->finalize = foundry_flatpak_options_finalize;
  object_class->get_property = foundry_flatpak_options_get_property;
  object_class->set_property = foundry_flatpak_options_set_property;

  serializable_class->deserialize_property = foundry_flatpak_options_deserialize_property;

  g_object_class_install_property (object_class,
                                   PROP_CFLAGS,
                                   g_param_spec_string ("cflags",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CFLAGS_OVERRIDE,
                                   g_param_spec_boolean ("cflags-override",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CXXFLAGS,
                                   g_param_spec_string ("cxxflags",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CXXFLAGS_OVERRIDE,
                                   g_param_spec_boolean ("cxxflags-override",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CPPFLAGS,
                                   g_param_spec_string ("cppflags",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CPPFLAGS_OVERRIDE,
                                   g_param_spec_boolean ("cppflags-override",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_LDFLAGS,
                                   g_param_spec_string ("ldflags",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_LDFLAGS_OVERRIDE,
                                   g_param_spec_boolean ("ldflags-override",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_APPEND_PATH,
                                   g_param_spec_string ("append-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_PREPEND_PATH,
                                   g_param_spec_string ("prepend-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_APPEND_LD_LIBRARY_PATH,
                                   g_param_spec_string ("append-ld-library-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_PREPEND_LD_LIBRARY_PATH,
                                   g_param_spec_string ("prepend-ld-library-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_APPEND_PKG_CONFIG_PATH,
                                   g_param_spec_string ("append-pkg-config-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_PREPEND_PKG_CONFIG_PATH,
                                   g_param_spec_string ("prepend-pkg-config-path",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_PREFIX,
                                   g_param_spec_string ("prefix",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_LIBDIR,
                                   g_param_spec_string ("libdir",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_ENV,
                                   g_param_spec_boxed ("env",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_ARCH,
                                   g_param_spec_object ("arch",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_ARCH_OPTIONS,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_BUILD_ARGS,
                                   g_param_spec_boxed ("build-args",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_TEST_ARGS,
                                   g_param_spec_boxed ("test-args",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CONFIG_OPTS,
                                   g_param_spec_boxed ("config-opts",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (object_class,
                                   PROP_SECRET_OPTS,
                                   g_param_spec_boxed ("secret-opts",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (object_class,
                                   PROP_SECRET_ENV,
                                   g_param_spec_boxed ("secret-env",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (object_class,
                                   PROP_MAKE_ARGS,
                                   g_param_spec_boxed ("make-args",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_MAKE_INSTALL_ARGS,
                                   g_param_spec_boxed ("make-install-args",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_STRIP,
                                   g_param_spec_boolean ("strip",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_DEBUGINFO,
                                   g_param_spec_boolean ("no-debuginfo",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_DEBUGINFO_COMPRESSION,
                                   g_param_spec_boolean ("no-debuginfo-compression",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void
foundry_flatpak_options_init (FoundryFlatpakOptions *self)
{
}

/**
 * foundry_flatpak_options_dup_build_args:
 * @self: a [class@Foundry.FlatpakOptions]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_options_dup_build_args (FoundryFlatpakOptions *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_OPTIONS (self), NULL);

  return g_strdupv (self->build_args);
}

char *
foundry_flatpak_options_dup_append_path (FoundryFlatpakOptions *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_OPTIONS (self), NULL);

  return g_strdup (self->append_path);
}

char *
foundry_flatpak_options_dup_prepend_path (FoundryFlatpakOptions *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_OPTIONS (self), NULL);

  return g_strdup (self->prepend_path);
}
