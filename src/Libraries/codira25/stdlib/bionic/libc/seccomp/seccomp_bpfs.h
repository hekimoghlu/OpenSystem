/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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

#include <stddef.h>
#include <linux/seccomp.h>

extern const struct sock_filter arm_app_filter[];
extern const size_t arm_app_filter_size;
extern const struct sock_filter arm_app_zygote_filter[];
extern const size_t arm_app_zygote_filter_size;
extern const struct sock_filter arm_system_filter[];
extern const size_t arm_system_filter_size;

extern const struct sock_filter arm64_app_filter[];
extern const size_t arm64_app_filter_size;
extern const struct sock_filter arm64_app_zygote_filter[];
extern const size_t arm64_app_zygote_filter_size;
extern const struct sock_filter arm64_system_filter[];
extern const size_t arm64_system_filter_size;

extern const struct sock_filter riscv64_app_filter[];
extern const size_t riscv64_app_filter_size;
extern const struct sock_filter riscv64_app_zygote_filter[];
extern const size_t riscv64_app_zygote_filter_size;
extern const struct sock_filter riscv64_system_filter[];
extern const size_t riscv64_system_filter_size;

extern const struct sock_filter x86_app_filter[];
extern const size_t x86_app_filter_size;
extern const struct sock_filter x86_app_zygote_filter[];
extern const size_t x86_app_zygote_filter_size;
extern const struct sock_filter x86_system_filter[];
extern const size_t x86_system_filter_size;

extern const struct sock_filter x86_64_app_filter[];
extern const size_t x86_64_app_filter_size;
extern const struct sock_filter x86_64_app_zygote_filter[];
extern const size_t x86_64_app_zygote_filter_size;
extern const struct sock_filter x86_64_system_filter[];
extern const size_t x86_64_system_filter_size;
