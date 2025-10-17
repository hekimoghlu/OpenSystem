/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 29, 2024.
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
#include "Utilities.h"
#include "SecTransform.h"
#include <sys/sysctl.h>
#include <syslog.h>
#include <dispatch/dispatch.h>

void MyDispatchAsync(dispatch_queue_t queue, void(^block)(void))
{
	fprintf(stderr, "Running job on queue %p\n", queue);
	dispatch_async(queue, block);
}



dispatch_queue_t MyDispatchQueueCreate(const char* name, dispatch_queue_attr_t attr)
{
	dispatch_queue_t result = dispatch_queue_create(name, attr);
	// fprintf(stderr, "Created queue %s as %p\n", name, result);
	return result;
}



static CFErrorRef CreateErrorRefCore(CFStringRef domain, int errorCode, CFStringRef format, va_list ap) __attribute__((format(__CFString__, 3, 0)));

static CFErrorRef CreateErrorRefCore(CFStringRef domain, int errorCode, CFStringRef format, va_list ap)
{
	CFStringRef str = CFStringCreateWithFormatAndArguments(NULL, NULL, format, ap);
	
	CFStringRef keys[] = {kCFErrorDescriptionKey};
	CFStringRef values[] = {str};
	
	CFErrorRef result = CFErrorCreateWithUserInfoKeysAndValues(NULL, domain, errorCode, (const void**) keys, (const void**) values, 1);
	CFReleaseNull(str);
	
	return result;
}



CFErrorRef CreateGenericErrorRef(CFStringRef domain, int errorCode, CFStringRef format, ...)
{
	va_list ap;
	va_start(ap, format);
	CFErrorRef ret = CreateErrorRefCore(domain, errorCode, format, ap);
	va_end(ap);
	return ret;
}



CFErrorRef CreateSecTransformErrorRef(int errorCode, CFStringRef format, ...)
{
	// create a CFError in the SecTransform error domain.  You can add an explanation, which is cool.
	va_list ap;
	va_start(ap, format);
	CFErrorRef ret = CreateErrorRefCore(kSecTransformErrorDomain, errorCode, format, ap);
	va_end(ap);
	return ret;
}



CFErrorRef CreateSecTransformErrorRefWithCFType(int errorCode, CFTypeRef message)
{
	CFStringRef keys[] = {kCFErrorLocalizedDescriptionKey};
	CFTypeRef values[] = {message};
	return CFErrorCreateWithUserInfoKeysAndValues(NULL, kSecTransformErrorDomain, errorCode, (const void**) keys, (const void**) values, 1);
}



CFTypeRef gAnnotatedRef = NULL;

CFTypeRef DebugRetain(const void* owner, CFTypeRef type)
{
	CFTypeRef result = CFRetainSafe(type);
	if (type == gAnnotatedRef)
	{
		fprintf(stderr, "Object %p was retained by object %p, count = %ld\n", type, owner, CFGetRetainCount(type));
	}
	
	return result;
}



void DebugRelease(const void* owner, CFTypeRef type)
{
	if (type == gAnnotatedRef)
	{
		fprintf(stderr, "Object %p was released by object %p, count = %ld\n", type, owner, CFGetRetainCount(type) - 1);
	}
	
	CFReleaseNull(type);
}

// Cribbed from _dispatch_bug and altered a bit
void transforms_bug(size_t line, long val)
{
    static dispatch_once_t pred;
    static char os_build[16];
    static void *last_seen;
    void *ra = __builtin_return_address(0);
    dispatch_once(&pred, ^{
#ifdef __APPLE__
        int mib[] = { CTL_KERN, KERN_OSVERSION };
        size_t bufsz = sizeof(os_build);
        sysctl(mib, 2, os_build, &bufsz, NULL, 0);
#else
        os_build[0] = '\0';
#endif
    });
    if (last_seen != ra) {
        last_seen = ra;
        syslog(LOG_NOTICE, "BUG in SecTransforms: %s - %p - %lu - %lu", os_build, last_seen, (unsigned long)line, val);
    }
}
