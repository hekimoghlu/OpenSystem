/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
/* test that the header doesn't implicitly depend on others */
#include <sys/resource_private.h>
#include <sys/resource.h>

#include <sys/types.h>
#include <unistd.h>

#include <darwintest.h>

T_GLOBAL_META(T_META_NAMESPACE("xnu.scheduler"),
    T_META_RADAR_COMPONENT_NAME("xnu"),
    T_META_RADAR_COMPONENT_VERSION("scheduler"),
    T_META_OWNER("chimene"),
    T_META_RUN_CONCURRENTLY(true),
    T_META_TAG_VM_PREFERRED);

T_DECL(unentitled_game_mode, "game mode bit shouldn't work unentitled")
{
	T_LOG("uid: %d", getuid());

	T_EXPECT_POSIX_FAILURE(setpriority(PRIO_DARWIN_GAME_MODE, 0, PRIO_DARWIN_GAME_MODE_ON),
	    EPERM, "setpriority(PRIO_DARWIN_GAME_MODE, 0, PRIO_DARWIN_GAME_MODE_ON)");
}

T_DECL(unentitled_game_mode_read_root, "game mode bit should be readable as root",
    T_META_ASROOT(true))
{
	T_LOG("uid: %d", getuid());

	T_ASSERT_POSIX_SUCCESS(getpriority(PRIO_DARWIN_GAME_MODE, 0),
	    "getpriority(PRIO_DARWIN_GAME_MODE)");
}

T_DECL(unentitled_game_mode_read_notroot, "game mode bit should not be readable as not root",
    T_META_ASROOT(false))
{
	T_LOG("uid: %d", getuid());

	T_EXPECT_POSIX_FAILURE(getpriority(PRIO_DARWIN_GAME_MODE, 0), EPERM,
	    "getpriority(PRIO_DARWIN_GAME_MODE)");
}
