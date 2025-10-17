/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_dyld_state.h"

#if PAS_OS(DARWIN)
#include <mach/mach_init.h>
#include <mach/task.h>
#include <mach/task_info.h>
#endif
#include "pas_log.h"

#if PAS_OS(DARWIN)
/* This is copied from dyld_process_info_internal.h
 
   FIXME: Stop doing it this way. Dyld should give us the is_libsystem_initialized API
   somehow. */
typedef struct {
    uint32_t                version;
    uint32_t                infoArrayCount;
    uint64_t                infoArray;
    uint64_t                notification;
    bool                    processDetachedFromSharedRegion;
    bool                    libSystemInitialized;
    /* There are more things after this, but we don't care. */
} dyld_all_image_infos_64;

static dyld_all_image_infos_64* all_image_infos;

bool pas_dyld_is_libsystem_initialized(void)
{
    dyld_all_image_infos_64* infos;

    infos = all_image_infos;
    
    if (!infos) {
        task_dyld_info_data_t task_dyld_info;
        mach_msg_type_number_t count;
        kern_return_t result;

        count = TASK_DYLD_INFO_COUNT;
        result = task_info(mach_task_self(),
                           TASK_DYLD_INFO,
                           (task_info_t)&task_dyld_info,
                           &count);
        PAS_ASSERT(result == KERN_SUCCESS);

        if (!task_dyld_info.all_image_info_addr)
            return false;

        infos = (dyld_all_image_infos_64*)task_dyld_info.all_image_info_addr;

        pas_fence();
        
        all_image_infos = infos;
    }

    return infos->libSystemInitialized;
}

#else

bool pas_dyld_is_libsystem_initialized(void)
{
    return true;
}

#endif

#endif /* LIBPAS_ENABLED */
