/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#define PLUGIN_CTX AUDIT_SYMBOL_NAME(plugin_ctx)
#define CALLBACK_CFUNC(func_name) AUDIT_SYMBOL_NAME(_python_plugin_audit_ ## func_name)

extern struct audit_plugin AUDIT_SYMBOL_NAME(python_audit);
static struct AuditPluginContext PLUGIN_CTX = { { NULL }, &AUDIT_SYMBOL_NAME(python_audit) };


static int
CALLBACK_CFUNC(open)(unsigned int version, sudo_conv_t conversation,
                         sudo_printf_t sudo_printf, char * const settings[],
                         char * const user_info[], int submit_optind,
                         char * const submit_argv[], char * const submit_envp[],
                         char * const plugin_options[], const char **errstr)
{
    return python_plugin_audit_open(&PLUGIN_CTX, version, conversation, sudo_printf,
        settings, user_info, submit_optind, submit_argv, submit_envp,
        plugin_options, errstr);
}

static void
CALLBACK_CFUNC(close)(int status_type, int status)
{
    python_plugin_audit_close(&PLUGIN_CTX, status_type, status);
}

static int
CALLBACK_CFUNC(accept)(const char *plugin_name, unsigned int plugin_type,
    char * const command_info[], char * const run_argv[],
    char * const run_envp[], const char **errstr)
{
    return python_plugin_audit_accept(&PLUGIN_CTX, plugin_name, plugin_type,
        command_info, run_argv, run_envp, errstr);
}

static int
CALLBACK_CFUNC(reject)(const char *plugin_name, unsigned int plugin_type,
    const char *audit_msg, char * const command_info[], const char **errstr)
{
    return python_plugin_audit_reject(&PLUGIN_CTX, plugin_name, plugin_type,
        audit_msg, command_info, errstr);
}

static int
CALLBACK_CFUNC(error)(const char *plugin_name, unsigned int plugin_type,
    const char *audit_msg, char * const command_info[], const char **errstr)
{
    return python_plugin_audit_error(&PLUGIN_CTX, plugin_name, plugin_type,
        audit_msg, command_info, errstr);
}

static int
CALLBACK_CFUNC(show_version)(int verbose)
{
    return python_plugin_audit_show_version(&PLUGIN_CTX, verbose);
}

struct audit_plugin AUDIT_SYMBOL_NAME(python_audit) = {
    SUDO_AUDIT_PLUGIN,
    SUDO_API_VERSION,
    CALLBACK_CFUNC(open),
    CALLBACK_CFUNC(close),
    CALLBACK_CFUNC(accept),
    CALLBACK_CFUNC(reject),
    CALLBACK_CFUNC(error),
    CALLBACK_CFUNC(show_version),
    NULL, /* register_hooks */
    NULL  /* deregister_hooks */
};

#undef PLUGIN_CTX
#undef CALLBACK_CFUNC
#undef AUDIT_SYMBOL_NAME
