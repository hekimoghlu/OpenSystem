/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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
#ifndef SUDO_PLUGIN_INT_H
#define SUDO_PLUGIN_INT_H

/*
 * All plugin structures start with a type and a version.
 */
struct generic_plugin {
    unsigned int type;
    unsigned int version;
    /* the rest depends on the type... */
};

typedef int (*sudo_conv_1_7_t)(int num_msgs,
    const struct sudo_conv_message msgs[], struct sudo_conv_reply replies[]);

/*
 * Backwards-compatible structures for API bumps.
 */
struct policy_plugin_1_0 {
    unsigned int type;
    unsigned int version;
    int (*open)(unsigned int version, sudo_conv_1_7_t conversation,
	sudo_printf_t sudo_plugin_printf, char * const settings[],
	char * const user_info[], char * const user_env[]);
    void (*close)(int exit_status, int error); /* wait status or error */
    int (*show_version)(int verbose);
    int (*check_policy)(int argc, char * const argv[],
	char *env_add[], char **command_info[],
	char **argv_out[], char **user_env_out[]);
    int (*list)(int argc, char * const argv[], int verbose,
	const char *user);
    int (*validate)(void);
    void (*invalidate)(int rmcred);
    int (*init_session)(struct passwd *pwd);
};
struct io_plugin_1_0 {
    unsigned int type;
    unsigned int version;
    int (*open)(unsigned int version, sudo_conv_1_7_t conversation,
        sudo_printf_t sudo_plugin_printf, char * const settings[],
        char * const user_info[], int argc, char * const argv[],
        char * const user_env[]);
    void (*close)(int exit_status, int error);
    int (*show_version)(int verbose);
    int (*log_ttyin)(const char *buf, unsigned int len);
    int (*log_ttyout)(const char *buf, unsigned int len);
    int (*log_stdin)(const char *buf, unsigned int len);
    int (*log_stdout)(const char *buf, unsigned int len);
    int (*log_stderr)(const char *buf, unsigned int len);
};
struct io_plugin_1_1 {
    unsigned int type;
    unsigned int version;
    int (*open)(unsigned int version, sudo_conv_1_7_t conversation,
	sudo_printf_t sudo_plugin_printf, char * const settings[],
	char * const user_info[], char * const command_info[],
	int argc, char * const argv[], char * const user_env[]);
    void (*close)(int exit_status, int error); /* wait status or error */
    int (*show_version)(int verbose);
    int (*log_ttyin)(const char *buf, unsigned int len);
    int (*log_ttyout)(const char *buf, unsigned int len);
    int (*log_stdin)(const char *buf, unsigned int len);
    int (*log_stdout)(const char *buf, unsigned int len);
    int (*log_stderr)(const char *buf, unsigned int len);
};

/*
 * Sudo plugin internals.
 */
struct plugin_container {
    TAILQ_ENTRY(plugin_container) entries;
    struct sudo_conf_debug_file_list *debug_files;
    char *name;
    char *path;
    char **options;
    void *handle;
    int debug_instance;
    union {
	struct generic_plugin *generic;
	struct policy_plugin *policy;
	struct policy_plugin_1_0 *policy_1_0;
	struct io_plugin *io;
	struct io_plugin_1_0 *io_1_0;
	struct io_plugin_1_1 *io_1_1;
	struct audit_plugin *audit;
	struct approval_plugin *approval;
    } u;
};
TAILQ_HEAD(plugin_container_list, plugin_container);

/*
 * Private implementation of struct sudo_plugin_event.
 */
struct sudo_plugin_event_int {
    struct sudo_event private;		/* must be first */
    int debug_instance;			/* plugin's debug instance */
    void *closure;			/* actual user closure */
    sudo_ev_callback_t callback;	/* actual user callback */
    struct sudo_plugin_event public;	/* user-visible portion */
};

extern struct plugin_container policy_plugin;
extern struct plugin_container_list io_plugins;
extern struct plugin_container_list audit_plugins;
extern struct plugin_container_list approval_plugins;

int sudo_conversation(int num_msgs, const struct sudo_conv_message msgs[],
    struct sudo_conv_reply replies[], struct sudo_conv_callback *callback);
int sudo_conversation_1_7(int num_msgs, const struct sudo_conv_message msgs[],
    struct sudo_conv_reply replies[]);
int sudo_conversation_printf(int msg_type, const char *fmt, ...);

bool sudo_load_plugins(void);

#endif /* SUDO_PLUGIN_INT_H */
