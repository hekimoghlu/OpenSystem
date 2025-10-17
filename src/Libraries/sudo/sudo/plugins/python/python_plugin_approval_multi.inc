/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
#define PLUGIN_CTX APPROVAL_SYMBOL_NAME(plugin_ctx)
#define CALLBACK_CFUNC(func_name) APPROVAL_SYMBOL_NAME(_python_plugin_approval_ ## func_name)

extern struct approval_plugin APPROVAL_SYMBOL_NAME(python_approval);
static struct ApprovalPluginContext PLUGIN_CTX = { { NULL }, &APPROVAL_SYMBOL_NAME(python_approval) };


static int
CALLBACK_CFUNC(open)(unsigned int version, sudo_conv_t conversation,
    sudo_printf_t sudo_printf, char * const settings[],
    char * const user_info[], int submit_optind,
    char * const submit_argv[], char * const submit_envp[],
    char * const plugin_options[], const char **errstr)
{
    return python_plugin_approval_open(&PLUGIN_CTX, version, conversation,
        sudo_printf, settings, user_info, submit_optind, submit_argv,
        submit_envp, plugin_options, errstr);
}

static void
CALLBACK_CFUNC(close)(void)
{
    python_plugin_approval_close(&PLUGIN_CTX);
}

static int
CALLBACK_CFUNC(check)(char * const command_info[], char * const run_argv[],
    char * const run_envp[], const char **errstr)
{
    return python_plugin_approval_check(&PLUGIN_CTX, command_info, run_argv,
        run_envp, errstr);
}

static int
CALLBACK_CFUNC(show_version)(int verbose)
{
    return python_plugin_approval_show_version(&PLUGIN_CTX, verbose);
}

struct approval_plugin APPROVAL_SYMBOL_NAME(python_approval) = {
    SUDO_APPROVAL_PLUGIN,
    SUDO_API_VERSION,
    CALLBACK_CFUNC(open),
    CALLBACK_CFUNC(close),
    CALLBACK_CFUNC(check),
    CALLBACK_CFUNC(show_version)
};

#undef PLUGIN_CTX
#undef CALLBACK_CFUNC
#undef APPROVAL_SYMBOL_NAME
