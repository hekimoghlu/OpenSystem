/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include "python_plugin_common.h"

static struct PluginContext plugin_ctx;

extern struct sudoers_group_plugin group_plugin;

#define PY_GROUP_PLUGIN_VERSION SUDO_API_MKVERSION(1, 0)

#define CALLBACK_PLUGINFUNC(func_name) group_plugin.func_name
#define CALLBACK_CFUNC(func_name) python_plugin_group_ ## func_name

// This also verifies compile time that the name matches the sudo plugin API.
#define CALLBACK_PYNAME(func_name) ((void)CALLBACK_PLUGINFUNC(func_name), #func_name)


static int
python_plugin_group_init(int version, sudo_printf_t sudo_printf, char *const plugin_options[])
{
    debug_decl(python_plugin_group_init, PYTHON_DEBUG_CALLBACKS);

    if (version < SUDO_API_MKVERSION(1, 0)) {
        sudo_printf(SUDO_CONV_ERROR_MSG,
                    "Error: Python group plugin requires at least plugin API version 1.0\n");
        debug_return_int(SUDO_RC_ERROR);
    }

    int rc = SUDO_RC_ERROR;

    rc = python_plugin_register_logging(NULL, sudo_printf, NULL);
    if (rc != SUDO_RC_OK)
        debug_return_int(rc);

    rc = python_plugin_init(&plugin_ctx, plugin_options, (unsigned int)version);
    if (rc != SUDO_RC_OK)
        debug_return_int(rc);

    PyObject *py_version = NULL,
             *py_plugin_options = NULL,
             *py_kwargs = NULL;

    if ((py_kwargs = PyDict_New()) == NULL ||
        (py_version = py_create_version(PY_GROUP_PLUGIN_VERSION)) == NULL ||
        (py_plugin_options = py_str_array_to_tuple(plugin_options)) == NULL ||
        PyDict_SetItemString(py_kwargs, "args", py_plugin_options) != 0 ||
        PyDict_SetItemString(py_kwargs, "version", py_version))
    {
        py_log_last_error("Failed to construct arguments for plugin constructor call.");
        rc = SUDO_RC_ERROR;
    } else {
        rc = python_plugin_construct_custom(&plugin_ctx, py_kwargs);
    }

    Py_XDECREF(py_version);
    Py_XDECREF(py_plugin_options);
    Py_XDECREF(py_kwargs);
    debug_return_int(rc);
}

static void
python_plugin_group_cleanup(void)
{
    debug_decl(python_plugin_group_cleanup, PYTHON_DEBUG_CALLBACKS);
    PyThreadState_Swap(plugin_ctx.py_interpreter);
    python_plugin_deinit(&plugin_ctx);
}

static int
python_plugin_group_query(const char *user, const char *group, const struct passwd *pwd)
{
    debug_decl(python_plugin_group_query, PYTHON_DEBUG_CALLBACKS);

    PyThreadState_Swap(plugin_ctx.py_interpreter);

    PyObject *py_pwd = py_from_passwd(pwd);
    if (py_pwd == NULL) {
        debug_return_int(SUDO_RC_ERROR);
    }

    int rc = python_plugin_api_rc_call(&plugin_ctx, CALLBACK_PYNAME(query),
                                       Py_BuildValue("(zzO)", user, group, py_pwd));
    Py_XDECREF(py_pwd);

    debug_return_int(rc);
}

sudo_dso_public struct sudoers_group_plugin group_plugin = {
    GROUP_API_VERSION,
    CALLBACK_CFUNC(init),
    CALLBACK_CFUNC(cleanup),
    CALLBACK_CFUNC(query)
};
