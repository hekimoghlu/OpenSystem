/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#define PLUGIN_CTX IO_SYMBOL_NAME(plugin_ctx)
#define CALLBACK_CFUNC(func_name) IO_SYMBOL_NAME(_python_plugin_io_ ## func_name)

extern struct io_plugin IO_SYMBOL_NAME(python_io);
static struct IOPluginContext PLUGIN_CTX = { { NULL }, &IO_SYMBOL_NAME(python_io) };

static int
CALLBACK_CFUNC(open)(
    unsigned int version, sudo_conv_t conversation,
    sudo_printf_t sudo_printf, char * const settings[],
    char * const user_info[], char * const command_info[],
    int argc, char * const argv[], char * const user_env[],
    char * const plugin_options[], const char **errstr)
{
    return python_plugin_io_open(&PLUGIN_CTX, version, conversation,
        sudo_printf, settings, user_info, command_info, argc, argv, user_env, plugin_options, errstr);
}

static void
CALLBACK_CFUNC(close)(int exit_status, int error)
{
    python_plugin_io_close(&PLUGIN_CTX, exit_status, error);
}

static int
CALLBACK_CFUNC(show_version)(int verbose)
{
    return python_plugin_io_show_version(&PLUGIN_CTX, verbose);
}

static int
CALLBACK_CFUNC(log_ttyin)(const char *buf, unsigned int len, const char **errstr)
{
    return python_plugin_io_log_ttyin(&PLUGIN_CTX, buf, len, errstr);
}

static int
CALLBACK_CFUNC(log_ttyout)(const char *buf, unsigned int len, const char **errstr)
{
    return python_plugin_io_log_ttyout(&PLUGIN_CTX, buf, len, errstr);
}

static int
CALLBACK_CFUNC(log_stdin)(const char *buf, unsigned int len, const char **errstr)
{
    return python_plugin_io_log_stdin(&PLUGIN_CTX, buf, len, errstr);
}

static int
CALLBACK_CFUNC(log_stdout)(const char *buf, unsigned int len, const char **errstr)
{
    return python_plugin_io_log_stdout(&PLUGIN_CTX, buf, len, errstr);
}

static int
CALLBACK_CFUNC(log_stderr)(const char *buf, unsigned int len, const char **errstr)
{
    return python_plugin_io_log_stderr(&PLUGIN_CTX, buf, len, errstr);
}

static int
CALLBACK_CFUNC(change_winsize)(unsigned int line, unsigned int cols, const char **errstr)
{
    return python_plugin_io_change_winsize(&PLUGIN_CTX, line, cols, errstr);
}

static int
CALLBACK_CFUNC(log_suspend)(int signo, const char **errstr)
{
    return python_plugin_io_log_suspend(&PLUGIN_CTX, signo, errstr);
}

struct io_plugin IO_SYMBOL_NAME(python_io) = {
    SUDO_IO_PLUGIN,
    SUDO_API_VERSION,
    CALLBACK_CFUNC(open),
    CALLBACK_CFUNC(close),
    CALLBACK_CFUNC(show_version),
    CALLBACK_CFUNC(log_ttyin),
    CALLBACK_CFUNC(log_ttyout),
    CALLBACK_CFUNC(log_stdin),
    CALLBACK_CFUNC(log_stdout),
    CALLBACK_CFUNC(log_stderr),
    NULL, // register_hooks,
    NULL, // deregister_hooks,
    CALLBACK_CFUNC(change_winsize),
    CALLBACK_CFUNC(log_suspend),
    NULL // event_alloc
};

#undef PLUGIN_CTX
#undef CALLBACK_CFUNC
#undef IO_SYMBOL_NAME
