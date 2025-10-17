/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#import <Foundation/Foundation.h>
#import <Foundation/NSKeyedArchiver_Private.h>
#import <CoreFoundation/CoreFoundation.h>
#import <AssertMacros.h>
#import <Security/Security.h>
#import "HeimCredCoder.h"
#import "common.h"
#import "gsscred.h"

@implementation HeimCredDecoder

static uuid_t boolean_true =  { 0xED, 0x43, 0xE0, 0xE6, 0x20, 0x15, 0x40, 0x43, 0xA9, 0x0C, 0xE7, 0x30, 0x6C, 0x9D, 0xED, 0x6B };
static uuid_t boolean_false = {  0x48, 0xD3, 0x95, 0xCD, 0x37, 0xC7, 0x41, 0x76, 0xA0, 0x97, 0x9B, 0x44, 0x9E, 0x2C, 0x10, 0x75 };


+ (CFTypeRef)copyNS2CF:(id)ns
{
    if (!ns)
	return NULL;
    
    if ([ns isKindOfClass:[NSString class]] || [ns isKindOfClass:[NSData class]] || [ns isKindOfClass:[NSNumber class]] || [ns isKindOfClass:[NSDate class]]) {
	return (CFTypeRef)CFBridgingRetain([ns copy]);
    }
    
    if ([ns isKindOfClass:[NSUUID class]]) {
        union {
	    CFUUIDBytes bytes;
	    uuid_t uuid;
	} u;
	
	[ns getUUIDBytes:u.uuid];
	
	if (uuid_compare(u.uuid, boolean_false) == 0)
	    return kCFBooleanFalse;
	else if (uuid_compare(u.uuid, boolean_true) == 0)
	    return kCFBooleanTrue;
	
	return CFUUIDCreateFromUUIDBytes(NULL, u.bytes);
    }
    
    if ([ns isKindOfClass:[NSArray class]]) {
	CFIndex n, len = [ns count];
	CFMutableArrayRef array = CFArrayCreateMutable(NULL, len, &kCFTypeArrayCallBacks);
	if (!array) return NULL;
	
	for (n = 0; n < len; n++) {
	    CFTypeRef obj = [HeimCredDecoder copyNS2CF:[ns objectAtIndex:n]];
	    if (obj) CFArrayAppendValue(array, obj);
	    CFRELEASE_NULL(obj);
	}
        return array;
    }
    if ([ns isKindOfClass:[NSDictionary class]]) {
        CFIndex len = [ns count];
	
	CFMutableDictionaryRef dict = CFDictionaryCreateMutable(NULL, len, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        if (!dict) return NULL;
	
	[ns enumerateKeysAndObjectsUsingBlock:^void(id key, id value, BOOL *stop) {
	    CFTypeRef cfkey = [HeimCredDecoder copyNS2CF:key];
	    CFTypeRef cfvalue = [HeimCredDecoder copyNS2CF:value];
	    if (cfkey && cfvalue)
		CFDictionarySetValue(dict, cfkey, cfvalue);
	    CFRELEASE_NULL(cfkey);
	    CFRELEASE_NULL(cfvalue);
	}];
	
	return dict;
    }
    return NULL;
}

static void
convertDict(const void *key, const void *value, void *context)
{
    NSMutableDictionary *dict = (__bridge NSMutableDictionary *)context;
    id nskey = [HeimCredDecoder copyCF2NS:key];
    id nsvalue = [HeimCredDecoder copyCF2NS:value];
    if (nskey && nsvalue)
	[dict setObject:nsvalue forKey:nskey];
}

+ (id)copyCF2NS:(CFTypeRef)cf
{
    if (!cf)
	return nil;
    
    CFTypeID type = CFGetTypeID(cf);
    
    if (type == CFBooleanGetTypeID()) {
	if (CFBooleanGetValue(cf))
	    return [[NSUUID alloc] initWithUUIDBytes:boolean_true];
	else
	    return [[NSUUID alloc] initWithUUIDBytes:boolean_false];
    }
    
    if (type == CFStringGetTypeID())
	return CFBridgingRelease(CFStringCreateCopy(NULL, cf));
    if (type == CFDataGetTypeID())
	return CFBridgingRelease(CFDataCreateCopy(NULL, cf));
    if (type == CFNumberGetTypeID())
	return (__bridge id)cf;
    if (type == CFDateGetTypeID())
	return (__bridge id)cf;
    
    if (type == CFUUIDGetTypeID()) {
        union {
	    CFUUIDBytes bytes;
	    uuid_t uuid;
	} u;
	
	u.bytes = CFUUIDGetUUIDBytes(cf);
	
	return [[NSUUID alloc] initWithUUIDBytes:u.uuid];
    }
    if (type == CFArrayGetTypeID()) {
	CFIndex n, len = CFArrayGetCount(cf);
	NSMutableArray *array = [[NSMutableArray alloc] initWithCapacity:len];
	if (!array) return NULL;
	
	for (n = 0; n < len; n++) {
	    id obj = [HeimCredDecoder copyCF2NS:CFArrayGetValueAtIndex(cf, n)];
	    if (obj) [array addObject:obj];
	}
        return array;
    }
    if (type == CFDictionaryGetTypeID()) {
        CFIndex len = CFDictionaryGetCount(cf);
	
        NSMutableDictionary *dict = [[NSMutableDictionary alloc] initWithCapacity:len];
        if (!dict) return NULL;
	
	CFDictionaryApplyFunction(cf, convertDict, (__bridge void *)dict);
	return dict;
    }
    return NULL;
}

+ (id) copyUnarchiveObjectWithFileSecureEncoding:(NSString *)path
{
    @autoreleasepool {
	NSData *clearText = NULL;
	NSData *data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:NULL];
	if (data == nil)
	    return NULL;
	
	clearText = HeimCredGlobalCTX.decryptData(data);
	if (clearText == NULL)
	    return NULL;
	
	HeimCredDecoder *decoder = [[HeimCredDecoder alloc] initForReadingFromData:(NSData *)clearText error:NULL];
	if (decoder == nil)
	    return NULL;
	
	id obj = [decoder decodeObjectOfClasses:[HeimCredDecoder allowedClasses] forKey:NSKeyedArchiveRootObjectKey];
	[decoder finishDecoding];
	
	return obj;
    }
}

+ (void)archiveRootObject:(id)object toFile:(NSString *)archiveFile
{
    @autoreleasepool {
	os_log_info(GSSOSLog(), "Save credentials to disk");
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:object requiringSecureCoding:YES error:&error];
	if (data == nil) {
	    os_log_error(GSSOSLog(), "Failed to archive credentials: %{public}@", error);
	    return;
	}
	
	NSData *encText = HeimCredGlobalCTX.encryptData(data);
	if (encText == NULL) {
	    os_log_error(GSSOSLog(), "Failed to wrap credentials");
	    [[NSFileManager defaultManager] removeItemAtPath:archiveFile error:NULL];
	    return;
	}

	NSDictionary *attributes = @{NSFilePosixPermissions:@(S_IRUSR|S_IWUSR)};
	BOOL worked = [[NSFileManager defaultManager] createFileAtPath:archiveFile contents:encText attributes:attributes];
	if (!worked) {
	    os_log_error(GSSOSLog(), "Error writing credentials to disk");
	}
    }
}

- (NSSet *)allowedClasses
{
    return [HeimCredDecoder allowedClasses];
}

+ (NSSet *)allowedClasses
{
    static dispatch_once_t onceToken;
    static NSSet *_set;
    dispatch_once(&onceToken, ^{
	_set = [[NSSet alloc] initWithObjects:[NSMutableDictionary class], [NSDictionary class],
		[NSMutableArray class], [NSArray class],
		[NSMutableString class], [NSString class],
		[NSNumber class],
		[NSUUID class],
		[NSMutableData class], [NSData class], [NSDate class],
		nil];
    });
    return _set;
}

@end
