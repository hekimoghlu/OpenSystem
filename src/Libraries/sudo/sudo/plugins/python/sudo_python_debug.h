/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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
#ifndef SUDO_PYTHON_DEBUG_H
#define SUDO_PYTHON_DEBUG_H

#include "sudo_debug.h"

/*
 * Sudo python plugin debug subsystems.
 * Note that python_subsystem_ids[] is filled in at debug registration time.
 */
extern int python_subsystem_ids[];
#define PYTHON_DEBUG_PY_CALLS    (python_subsystem_ids[0])
#define PYTHON_DEBUG_C_CALLS     (python_subsystem_ids[1])
#define PYTHON_DEBUG_PLUGIN_LOAD (python_subsystem_ids[2])
#define PYTHON_DEBUG_CALLBACKS   (python_subsystem_ids[3])
#define PYTHON_DEBUG_INTERNAL    (python_subsystem_ids[4])
#define PYTHON_DEBUG_PLUGIN      (python_subsystem_ids[5])

bool python_debug_parse_flags(struct sudo_conf_debug_file_list *debug_files, const char *entry);
bool python_debug_register(const char *program, struct sudo_conf_debug_file_list *debug_files);
void python_debug_deregister(void);

#define debug_return_ptr_pynone \
    do { \
        Py_INCREF(Py_None); \
        debug_return_ptr(Py_None); \
    } while(0)

#endif /* SUDO_PYTHON_DEBUG_H */
