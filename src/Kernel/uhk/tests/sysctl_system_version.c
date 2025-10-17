/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include <darwintest.h>
#include <sys/sysctl.h>

T_DECL(sysctl_osreleasetype_nowrite,
    "ensure the osreleasetype sysctl is not writeable by normal processes", T_META_TAG_VM_NOT_PREFERRED)
{
	char nice_try[32] = "FactoryToAvoidSandbox!";
	int ret = sysctlbyname("kern.osreleasetype", NULL, NULL, nice_try,
	    sizeof(nice_try));
	T_ASSERT_POSIX_FAILURE(ret, EPERM, "try to set kern.osreleasetype sysctl");
}

T_DECL(sysctl_osreleasetype_exists, "ensure the osreleasetype sysctl exists", T_META_TAG_VM_NOT_PREFERRED)
{
	char release_type[64] = "";
	size_t release_type_size = sizeof(release_type);
	int ret = sysctlbyname("kern.osreleasetype", release_type,
	    &release_type_size, NULL, 0);
	T_ASSERT_POSIX_SUCCESS(ret, "kern.osreleasetype sysctl");
	T_LOG("kern.osreleasetype = %s", release_type);
}
