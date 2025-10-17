/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#import "config.h"
#import "WKUtilities.h"

#if PLATFORM(IOS_FAMILY)

#import <wtf/Assertions.h>

const CFArrayCallBacks WKCollectionArrayCallBacks = { 0, WKCollectionRetain, WKCollectionRelease, NULL, NULL };
const CFSetCallBacks WKCollectionSetCallBacks = { 0, WKCollectionRetain, WKCollectionRelease, NULL, NULL, NULL };

const void *WKCollectionRetain (CFAllocatorRef allocator, const void *value)
{
    UNUSED_PARAM(allocator);
    return WAKRetain (value);
}

const void *WAKRetain(const void *o)
{
    WAKObjectRef object = (WAKObjectRef)(uintptr_t)o;
    
    object->referenceCount++;
    
    return object;
}

void WKCollectionRelease (CFAllocatorRef allocator, const void *value)
{
    UNUSED_PARAM(allocator);
    WAKRelease (value);
}

void WAKRelease(const void *o)
{
    WAKObjectRef object = (WAKObjectRef)(uintptr_t)o;

    if (object->referenceCount == 0) {
        WKError ("attempt to release invalid object");
        return;
    }
    
    object->referenceCount--;

    if (object->referenceCount == 0) {
        const WKClassInfo *info = object->classInfo;
        while (info) {
            if (info->dealloc)
                info->dealloc ((void *)(uintptr_t)object);
            info = info->parent;
        }
    }
}

static void WAKObjectDealloc(WAKObjectRef v)
{
    free (v);
}

WKClassInfo WAKObjectClass = { 0, "WAKObject", WAKObjectDealloc };

const void *WKCreateObjectWithSize (size_t size, WKClassInfo *info)
{
    WAKObjectRef object = (WAKObjectRef)calloc(size, 1);
    if (!object)
        return 0;

    object->classInfo = info;
    
    WAKRetain(object);
    
    return object;
}

WTF_ATTRIBUTE_PRINTF(4, 5)
void WKReportError(const char *file, int line, const char *function, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "%s:%d %s:  ", file, line, function);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
}

CFIndex WKArrayIndexOfValue (CFArrayRef array, const void *value)
{
    CFIndex i, count, index = -1;

    count = CFArrayGetCount (array);
    for (i = 0; i < count; i++) {
        if (CFArrayGetValueAtIndex (array, i) == value) {
            index = i;
            break;
        }
    }
    
    return index;
}

WKClassInfo *WKGetClassInfo(WAKObjectRef object)
{
    return object->classInfo;
}

#endif // PLATFORM(IOS_FAMILY)
