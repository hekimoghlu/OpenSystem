/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#ifndef __DISKARBITRATIOND_DABASE__
#define __DISKARBITRATIOND_DABASE__

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <SystemConfiguration/SystemConfiguration.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define ___kCFUUIDNull CFUUIDGetConstantUUIDWithBytes( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

#define ___EDIRTY EILSEQ

#define ___FS_DEFAULT_DIR "/Library/Filesystems"

#define ___PREFS_DEFAULT_DIR "/Library/Preferences/SystemConfiguration"

typedef char ___io_path_t[1024];

__private_extern__ int             ___isautofs( const char * path );
__private_extern__ int             ___mkdir( const char * path, mode_t mode );
__private_extern__ void            ___vproc_transaction_begin( void );
__private_extern__ void            ___vproc_transaction_end( void );
__private_extern__ const void *    ___CFArrayGetValue( CFArrayRef array, const void * value );
__private_extern__ void            ___CFArrayIntersect( CFMutableArrayRef array1, CFArrayRef array2 );
__private_extern__ CFStringRef     ___CFBundleCopyLocalizedStringInDirectory( CFURLRef bundleURL, CFStringRef key, CFStringRef value, CFStringRef table );
__private_extern__ CFURLRef        ___CFBundleCopyResourceURLInDirectory( CFURLRef bundleURL, CFStringRef resourcePath );
__private_extern__ CFDataRef       ___CFDataCreateFromString( CFAllocatorRef allocator, CFStringRef string );
__private_extern__ CFDictionaryRef ___CFDictionaryCreateFromXMLString( CFAllocatorRef allocator, CFStringRef string );
__private_extern__ const void *    ___CFDictionaryGetAnyValue( CFDictionaryRef dictionary );
__private_extern__ char *          ___CFStringCreateCStringWithFormatAndArguments( const char * format, va_list arguments );
__private_extern__ Boolean         ___CFStringGetCString( CFStringRef string, char * buffer, CFIndex length );
__private_extern__ void            ___CFStringInsertFormat( CFMutableStringRef string, CFIndex index, CFStringRef format, ... );
__private_extern__ void            ___CFStringInsertFormatAndArguments( CFMutableStringRef string, CFIndex index, CFStringRef format, va_list arguments );
__private_extern__ void            ___CFStringPad( CFMutableStringRef string, CFStringRef pad, CFIndex length, CFIndex index );
__private_extern__ CFUUIDRef       ___CFUUIDCreateFromName( CFAllocatorRef allocator, CFUUIDRef space, CFDataRef name );
__private_extern__ CFUUIDRef       ___CFUUIDCreateFromString( CFAllocatorRef allocator, CFStringRef string );
__private_extern__ CFStringRef     ___CFURLCopyRawDeviceFileSystemPath( CFURLRef url, CFURLPathStyle pathStyle );
__private_extern__ kern_return_t   ___IORegistryEntryGetPath( io_registry_entry_t entry, const io_name_t plane, ___io_path_t path );
#if TARGET_OS_OSX
__private_extern__ CFArrayRef      ___SCDynamicStoreCopyConsoleInformation( SCDynamicStoreRef store );
__private_extern__ CFStringRef     ___SCDynamicStoreCopyConsoleUser( SCDynamicStoreRef store, uid_t * uid, gid_t * gid );
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DABASE__ */
