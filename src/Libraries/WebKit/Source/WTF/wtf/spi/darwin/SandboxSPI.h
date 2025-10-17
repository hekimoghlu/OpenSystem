/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#pragma once

#if OS(DARWIN)

#import <sandbox.h>

#if USE(APPLE_INTERNAL_SDK)
#import <sandbox/private.h>
#else
enum sandbox_filter_type {
    SANDBOX_FILTER_NONE,
    SANDBOX_FILTER_PATH,
    SANDBOX_FILTER_GLOBAL_NAME = 2,
    SANDBOX_FILTER_PREFERENCE_DOMAIN = 6,
    SANDBOX_FILTER_XPC_SERVICE_NAME = 12,
    SANDBOX_FILTER_IOKIT_CONNECTION,
    SANDBOX_FILTER_SYSCALL_NUMBER,
};

#define SANDBOX_NAMED_EXTERNAL 0x0003
#endif

WTF_EXTERN_C_BEGIN

typedef struct {
    char* builtin;
    unsigned char* data;
    size_t size;
} *sandbox_profile_t;

typedef struct {
    const char **params;
    size_t size;
    size_t available;
} *sandbox_params_t;

extern const char *const APP_SANDBOX_READ;
extern const char *const APP_SANDBOX_READ_WRITE;
extern const enum sandbox_filter_type SANDBOX_CHECK_NO_REPORT;

extern const uint32_t SANDBOX_EXTENSION_NO_REPORT;
extern const uint32_t SANDBOX_EXTENSION_CANONICAL;
extern const uint32_t SANDBOX_EXTENSION_USER_INTENT;

char *sandbox_extension_issue_file(const char *extension_class, const char *path, uint32_t flags);
char *sandbox_extension_issue_generic(const char *extension_class, uint32_t flags);
char *sandbox_extension_issue_file_to_process(const char *extension_class, const char *path, uint32_t flags, audit_token_t);
char *sandbox_extension_issue_mach_to_process(const char *extension_class, const char *name, uint32_t flags, audit_token_t);
char *sandbox_extension_issue_mach(const char *extension_class, const char *name, uint32_t flags);
int sandbox_check(pid_t, const char *operation, enum sandbox_filter_type, ...);
int sandbox_check_by_audit_token(audit_token_t, const char *operation, enum sandbox_filter_type, ...);
int sandbox_container_path_for_pid(pid_t, char *buffer, size_t bufsize);
int sandbox_extension_release(int64_t extension_handle);
int sandbox_init_with_parameters(const char *profile, uint64_t flags, const char *const parameters[], char **errorbuf);
int64_t sandbox_extension_consume(const char *extension_token);
sandbox_params_t sandbox_create_params(void);
int sandbox_set_param(sandbox_params_t, const char *key, const char *value);
void sandbox_free_params(sandbox_params_t);
sandbox_profile_t sandbox_compile_file(const char *path, sandbox_params_t, char **error);
sandbox_profile_t sandbox_compile_string(const char *data, sandbox_params_t, char **error);
void sandbox_free_profile(sandbox_profile_t);
int sandbox_apply(sandbox_profile_t);

char *sandbox_extension_issue_iokit_registry_entry_class_to_process(const char *extension_class, const char *registry_entry_class, uint32_t flags, audit_token_t);
char *sandbox_extension_issue_iokit_registry_entry_class(const char *extension_class, const char *registry_entry_class, uint32_t flags);

bool sandbox_enable_state_flag(const char *varname, audit_token_t);

WTF_EXTERN_C_END

#endif // OS(DARWIN)
