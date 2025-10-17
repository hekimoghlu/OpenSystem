/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef SUDO_PYTHON_PLUGIN_COMMON_H
#define SUDO_PYTHON_PLUGIN_COMMON_H

#include "pyhelpers.h"

struct PluginContext {
    PyThreadState *py_interpreter;
    PyObject *py_module;
    PyObject *py_class;
    PyObject *py_instance;
    int call_close;
    unsigned int sudo_api_version;
    char *plugin_path;

    // We use this to let the error string live until sudo and the audit plugins
    // are using it.
    char *callback_error;
};

int python_plugin_register_logging(sudo_conv_t conversation, sudo_printf_t sudo_printf, char * const settings[]);

int python_plugin_init(struct PluginContext *plugin_ctx, char * const plugin_options[], unsigned int version);

int python_plugin_construct_custom(struct PluginContext *plugin_ctx, PyObject *py_kwargs);

PyObject *python_plugin_construct_args(unsigned int version, char *const settings[],
    char *const user_info[], char *const user_env[], char *const plugin_options[]);

int python_plugin_construct(struct PluginContext *plugin_ctx, unsigned int version,
    char *const settings[], char *const user_info[],
    char *const user_env[], char *const plugin_options[]);

void python_plugin_deinit(struct PluginContext *plugin_ctx);

int python_plugin_show_version(struct PluginContext *plugin_ctx,
                               const char *python_callback_name, int isVerbose, unsigned int plugin_api_version, const char *plugin_api_name);

CPYCHECKER_STEALS_REFERENCE_TO_ARG(3)
void python_plugin_close(struct PluginContext *plugin_ctx, const char *callback_name,
                         PyObject *py_args);

CPYCHECKER_STEALS_REFERENCE_TO_ARG(3)
PyObject *python_plugin_api_call(struct PluginContext *plugin_ctx,
                                 const char *func_name, PyObject *py_args);

CPYCHECKER_STEALS_REFERENCE_TO_ARG(3)
int python_plugin_api_rc_call(struct PluginContext *plugin_ctx,
                              const char *func_name, PyObject *py_args);

int python_plugin_rc_to_int(PyObject *py_result);

void python_plugin_mark_callback_optional(struct PluginContext *plugin_ctx,
                                          const char *function_name, void **function);

const char *python_plugin_name(struct PluginContext *plugin_ctx);

// sets the callback error stored in plugin_ctx into "errstr" but only if API
// version is enough and "errstr" is valid
#define CALLBACK_SET_ERROR(plugin_ctx, errstr) \
    do { \
        if ((plugin_ctx)->sudo_api_version >= SUDO_API_MKVERSION(1, 15)) { \
            if (errstr != NULL) \
                *errstr = (plugin_ctx)->callback_error; \
        } \
    } while(0)

#endif // SUDO_PYTHON_PLUGIN_COMMON_H
