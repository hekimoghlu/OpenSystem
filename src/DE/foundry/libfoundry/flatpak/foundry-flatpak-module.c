/* foundry-flatpak-module.c
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

#include "foundry-flatpak-module.h"
#include "foundry-flatpak-modules.h"
#include "foundry-flatpak-options.h"
#include "foundry-flatpak-sources.h"

struct _FoundryFlatpakModule
{
  FoundryFlatpakSerializable   parent_instance;

  char                        *name;
  char                        *subdir;
  char                        *install_rule;
  char                        *test_rule;
  char                        *buildsystem;

  char                       **post_install;
  char                       **config_opts;
  char                       **secret_opts;
  char                       **secret_env;
  char                       **make_args;
  char                       **make_install_args;
  char                       **ensure_writable;
  char                       **only_arches;
  char                       **skip_arches;
  char                       **cleanup;
  char                       **cleanup_platform;
  char                       **build_commands;
  char                       **test_commands;

  FoundryFlatpakOptions       *build_options;
  FoundryFlatpakModules       *modules;
  FoundryFlatpakSources       *sources;

  guint                        disabled : 1;
  guint                        rm_configure : 1;
  guint                        no_autogen : 1;
  guint                        no_parallel_make : 1;
  guint                        no_make_install : 1;
  guint                        no_python_timestamp_fix : 1;
  guint                        cmake : 1;
  guint                        builddir : 1;
  guint                        run_tests : 1;
};

enum {
  PROP_0,
  PROP_BUILDDIR,
  PROP_BUILDSYSTEM,
  PROP_BUILD_COMMANDS,
  PROP_BUILD_OPTIONS,
  PROP_CLEANUP,
  PROP_CLEANUP_PLATFORM,
  PROP_CMAKE,
  PROP_CONFIG_OPTS,
  PROP_DISABLED,
  PROP_ENSURE_WRITABLE,
  PROP_INSTALL_RULE,
  PROP_MAKE_ARGS,
  PROP_MAKE_INSTALL_ARGS,
  PROP_MODULES,
  PROP_NAME,
  PROP_NO_AUTOGEN,
  PROP_NO_MAKE_INSTALL,
  PROP_NO_PARALLEL_MAKE,
  PROP_NO_PYTHON_TIMESTAMP_FIX,
  PROP_ONLY_ARCHES,
  PROP_POST_INSTALL,
  PROP_RM_CONFIGURE,
  PROP_RUN_TESTS,
  PROP_SECRET_ENV,
  PROP_SECRET_OPTS,
  PROP_SKIP_ARCHES,
  PROP_SOURCES,
  PROP_SUBDIR,
  PROP_TEST_COMMANDS,
  PROP_TEST_RULE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakModule, foundry_flatpak_module, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE)

static void
foundry_flatpak_module_finalize (GObject *object)
{
  FoundryFlatpakModule *self = (FoundryFlatpakModule *)object;

  g_clear_object (&self->build_options);
  g_clear_object (&self->modules);
  g_clear_object (&self->sources);

  g_clear_pointer (&self->buildsystem, g_free);
  g_clear_pointer (&self->install_rule, g_free);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->subdir, g_free);
  g_clear_pointer (&self->test_rule, g_free);

  g_clear_pointer (&self->build_commands, g_strfreev);
  g_clear_pointer (&self->cleanup, g_strfreev);
  g_clear_pointer (&self->cleanup_platform, g_strfreev);
  g_clear_pointer (&self->config_opts, g_strfreev);
  g_clear_pointer (&self->ensure_writable, g_strfreev);
  g_clear_pointer (&self->make_args, g_strfreev);
  g_clear_pointer (&self->make_install_args, g_strfreev);
  g_clear_pointer (&self->only_arches, g_strfreev);
  g_clear_pointer (&self->post_install, g_strfreev);
  g_clear_pointer (&self->secret_env, g_strfreev);
  g_clear_pointer (&self->secret_opts, g_strfreev);
  g_clear_pointer (&self->skip_arches, g_strfreev);
  g_clear_pointer (&self->test_commands, g_strfreev);

  G_OBJECT_CLASS (foundry_flatpak_module_parent_class)->finalize (object);
}

static void
foundry_flatpak_module_get_property (GObject    *object,
                                     guint       prop_id,
                                     GValue     *value,
                                     GParamSpec *pspec)
{
  FoundryFlatpakModule *self = FOUNDRY_FLATPAK_MODULE (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_set_string (value, self->name);
      break;

    case PROP_SUBDIR:
      g_value_set_string (value, self->subdir);
      break;

    case PROP_RM_CONFIGURE:
      g_value_set_boolean (value, self->rm_configure);
      break;

    case PROP_DISABLED:
      g_value_set_boolean (value, self->disabled);
      break;

    case PROP_NO_AUTOGEN:
      g_value_set_boolean (value, self->no_autogen);
      break;

    case PROP_NO_PARALLEL_MAKE:
      g_value_set_boolean (value, self->no_parallel_make);
      break;

    case PROP_NO_MAKE_INSTALL:
      g_value_set_boolean (value, self->no_make_install);
      break;

    case PROP_NO_PYTHON_TIMESTAMP_FIX:
      g_value_set_boolean (value, self->no_python_timestamp_fix);
      break;

    case PROP_CMAKE:
      g_value_set_boolean (value, self->cmake);
      break;

    case PROP_BUILDSYSTEM:
      g_value_set_string (value, self->buildsystem);
      break;

    case PROP_INSTALL_RULE:
      g_value_set_string (value, self->install_rule);
      break;

    case PROP_TEST_RULE:
      g_value_set_string (value, self->test_rule);
      break;

    case PROP_BUILDDIR:
      g_value_set_boolean (value, self->builddir);
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

    case PROP_ENSURE_WRITABLE:
      g_value_set_boxed (value, self->ensure_writable);
      break;

    case PROP_ONLY_ARCHES:
      g_value_set_boxed (value, self->only_arches);
      break;

    case PROP_SKIP_ARCHES:
      g_value_set_boxed (value, self->skip_arches);
      break;

    case PROP_POST_INSTALL:
      g_value_set_boxed (value, self->post_install);
      break;

    case PROP_BUILD_OPTIONS:
      g_value_set_object (value, self->build_options);
      break;

    case PROP_SOURCES:
      g_value_set_object (value, self->sources);
      break;

    case PROP_CLEANUP:
      g_value_set_boxed (value, self->cleanup);
      break;

    case PROP_CLEANUP_PLATFORM:
      g_value_set_boxed (value, self->cleanup_platform);
      break;

    case PROP_MODULES:
      g_value_set_object (value, self->modules);
      break;

    case PROP_BUILD_COMMANDS:
      g_value_set_boxed (value, self->build_commands);
      break;

    case PROP_TEST_COMMANDS:
      g_value_set_boxed (value, self->test_commands);
      break;

    case PROP_RUN_TESTS:
      g_value_set_boolean (value, self->run_tests);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_module_set_property (GObject      *object,
                                     guint         prop_id,
                                     const GValue *value,
                                     GParamSpec   *pspec)
{
  FoundryFlatpakModule *self = FOUNDRY_FLATPAK_MODULE (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_set_str (&self->name, g_value_get_string (value));
      break;

    case PROP_SUBDIR:
      g_set_str (&self->subdir, g_value_get_string (value));
      break;

    case PROP_RM_CONFIGURE:
      self->rm_configure = g_value_get_boolean (value);
      break;

    case PROP_DISABLED:
      self->disabled = g_value_get_boolean (value);
      break;

    case PROP_NO_AUTOGEN:
      self->no_autogen = g_value_get_boolean (value);
      break;

    case PROP_NO_PARALLEL_MAKE:
      self->no_parallel_make = g_value_get_boolean (value);
      break;

    case PROP_NO_MAKE_INSTALL:
      self->no_make_install = g_value_get_boolean (value);
      break;

    case PROP_NO_PYTHON_TIMESTAMP_FIX:
      self->no_python_timestamp_fix = g_value_get_boolean (value);
      break;

    case PROP_CMAKE:
      self->cmake = g_value_get_boolean (value);
      break;

    case PROP_BUILDSYSTEM:
      g_set_str (&self->buildsystem, g_value_get_string (value));
      break;

    case PROP_INSTALL_RULE:
      g_set_str (&self->install_rule, g_value_get_string (value));
      break;

    case PROP_TEST_RULE:
      g_set_str (&self->test_rule, g_value_get_string (value));
      break;

    case PROP_BUILDDIR:
      self->builddir = g_value_get_boolean (value);
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

    case PROP_ENSURE_WRITABLE:
      foundry_set_strv (&self->ensure_writable, g_value_get_boxed (value));
      break;

    case PROP_ONLY_ARCHES:
      foundry_set_strv (&self->only_arches, g_value_get_boxed (value));
      break;

    case PROP_SKIP_ARCHES:
      foundry_set_strv (&self->skip_arches, g_value_get_boxed (value));
      break;

    case PROP_POST_INSTALL:
      foundry_set_strv (&self->post_install, g_value_get_boxed (value));
      break;

    case PROP_BUILD_OPTIONS:
      g_set_object (&self->build_options, g_value_get_object (value));
      break;

    case PROP_SOURCES:
      g_set_object (&self->sources, g_value_get_object (value));
      break;

    case PROP_CLEANUP:
      foundry_set_strv (&self->cleanup, g_value_get_boxed (value));
      break;

    case PROP_CLEANUP_PLATFORM:
      foundry_set_strv (&self->cleanup_platform, g_value_get_boxed (value));
      break;

    case PROP_MODULES:
      g_set_object (&self->modules, g_value_get_object (value));
      break;

    case PROP_BUILD_COMMANDS:
      foundry_set_strv (&self->build_commands, g_value_get_boxed (value));
      break;

    case PROP_TEST_COMMANDS:
      foundry_set_strv (&self->test_commands, g_value_get_boxed (value));
      break;

    case PROP_RUN_TESTS:
      self->run_tests = g_value_get_boolean (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_module_class_init (FoundryFlatpakModuleClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_flatpak_module_finalize;
  object_class->get_property = foundry_flatpak_module_get_property;
  object_class->set_property = foundry_flatpak_module_set_property;

  g_object_class_install_property (object_class,
                                   PROP_NAME,
                                   g_param_spec_string ("name",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_SUBDIR,
                                   g_param_spec_string ("subdir",
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_RM_CONFIGURE,
                                   g_param_spec_boolean ("rm-configure",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_DISABLED,
                                   g_param_spec_boolean ("disabled",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_AUTOGEN,
                                   g_param_spec_boolean ("no-autogen",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_PARALLEL_MAKE,
                                   g_param_spec_boolean ("no-parallel-make",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_MAKE_INSTALL,
                                   g_param_spec_boolean ("no-make-install",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_NO_PYTHON_TIMESTAMP_FIX,
                                   g_param_spec_boolean ("no-python-timestamp-fix",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CMAKE,
                                   g_param_spec_boolean ("cmake",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS|G_PARAM_DEPRECATED));
  g_object_class_install_property (object_class,
                                   PROP_BUILDSYSTEM,
                                   g_param_spec_string ("buildsystem",
                                                         NULL,
                                                         NULL,
                                                         NULL,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_INSTALL_RULE,
                                   g_param_spec_string ("install-rule",
                                                         NULL,
                                                         NULL,
                                                         NULL,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_TEST_RULE,
                                   g_param_spec_string ("test-rule",
                                                         NULL,
                                                         NULL,
                                                         NULL,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_BUILDDIR,
                                   g_param_spec_boolean ("builddir",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_SOURCES,
                                   g_param_spec_object ("sources",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_SOURCES,
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
                                   PROP_ENSURE_WRITABLE,
                                   g_param_spec_boxed ("ensure-writable",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_ONLY_ARCHES,
                                   g_param_spec_boxed ("only-arches",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_SKIP_ARCHES,
                                   g_param_spec_boxed ("skip-arches",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_POST_INSTALL,
                                   g_param_spec_boxed ("post-install",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_BUILD_OPTIONS,
                                   g_param_spec_object ("build-options",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_OPTIONS,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CLEANUP,
                                   g_param_spec_boxed ("cleanup",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_CLEANUP_PLATFORM,
                                   g_param_spec_boxed ("cleanup-platform",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_MODULES,
                                   g_param_spec_object ("modules",
                                                        NULL,
                                                        NULL,
                                                        FOUNDRY_TYPE_FLATPAK_MODULES,
                                                        G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_BUILD_COMMANDS,
                                   g_param_spec_boxed ("build-commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_TEST_COMMANDS,
                                   g_param_spec_boxed ("test-commands",
                                                       NULL,
                                                       NULL,
                                                       G_TYPE_STRV,
                                                       G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property (object_class,
                                   PROP_RUN_TESTS,
                                   g_param_spec_boolean ("run-tests",
                                                         NULL,
                                                         NULL,
                                                         FALSE,
                                                         G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void
foundry_flatpak_module_init (FoundryFlatpakModule *self)
{
}

char *
foundry_flatpak_module_dup_name (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return g_strdup (self->name);
}

/**
 * foundry_flatpak_module_dup_modules:
 * @self: a [class@Foundry.FlatpakModule]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakModules *
foundry_flatpak_module_dup_modules (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return self->modules ? g_object_ref (self->modules) : NULL;
}

/**
 * foundry_flatpak_module_dup_build_options:
 * @self: a [class@Foundry.FlatpakModule]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakOptions *
foundry_flatpak_module_dup_build_options (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return self->build_options ? g_object_ref (self->build_options) : NULL;
}

char *
foundry_flatpak_module_dup_buildsystem (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return g_strdup (self->buildsystem);
}

/**
 * foundry_flatpak_module_dup_config_opts:
 * @self: a [class@Foundry.FlatpakModule]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_module_dup_config_opts (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return g_strdupv (self->config_opts);
}

/**
 * foundry_flatpak_module_dup_sources:
 * @self: a [class@Foundry.FlatpakModule]
 *
 * Returns: (transfer full) (nullable):
 */
FoundryFlatpakSources *
foundry_flatpak_module_dup_sources (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return self->sources ? g_object_ref (self->sources) : NULL;
}

/**
 * foundry_flatpak_module_dup_build_commands:
 * @self: a [class@Foundry.FlatpakModule]
 *
 * Returns: (transfer full) (nullable):
 */
char **
foundry_flatpak_module_dup_build_commands (FoundryFlatpakModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MODULE (self), NULL);

  return g_strdupv (self->build_commands);
}
