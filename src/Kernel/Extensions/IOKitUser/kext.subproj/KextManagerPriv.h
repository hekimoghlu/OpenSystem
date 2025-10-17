/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#ifndef __KEXTMANAGERPRIV_H__
#define __KEXTMANAGERPRIV_H__

#include <CoreFoundation/CoreFoundation.h>
#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE
#include <Security/Authorization.h>
#endif /* !TARGET_OS_IPHONE */

#include <sys/cdefs.h>

__BEGIN_DECLS

#define kKextLoadIdentifierKey   CFSTR("KextLoadIdentifier")
#define kKextLoadPathKey         CFSTR("KextLoadPath")
#define kKextLoadDependenciesKey CFSTR("KextLoadDependencyPaths")

#define kExtPathKey             CFSTR("ExtPath")
#define kExtEnabledKey          CFSTR("ExtEnabled")

CFArrayRef _KextManagerCreatePropertyValueArray(
    CFAllocatorRef allocator,
    CFStringRef    propertyKey);

/*
 * This is part of a private, entitled interface between kextd, which manages
 * the lifecycle of kernel and userspace driver extensions, and sysextd,
 * which manages the installation of third-party system extensions.
 * If you are not sysextd or kextd, you should not use these functions.
 * They are liable to change at any time.
 */

/* Validate an extension in-place on the filesystem. */
OSReturn _KextManagerValidateExtension(CFStringRef extPath);
/* Update an extension's enablement state. */
OSReturn _KextManagerUpdateExtension(CFStringRef extPath, bool extIsEnabled);
/* Ask kextd to stop an extension - it can say no if it is unable. */
OSReturn _KextManagerStopExtension(CFStringRef extPath);

__END_DECLS

#endif /* __KEXTMANAGERPRIV_H__ */
