/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#import "QuickLook.h"

#if USE(QUICK_LOOK)

#import "ResourceRequest.h"
#import <pal/spi/cocoa/NSFileManagerSPI.h>
#import <wtf/FileSystem.h>
#import <wtf/Lock.h>
#import <wtf/NeverDestroyed.h>

#import <pal/ios/QuickLookSoftLink.h>

namespace WebCore {

NSSet *QLPreviewGetSupportedMIMETypesSet()
{
    static NeverDestroyed<RetainPtr<NSSet>> set = PAL::softLink_QuickLook_QLPreviewGetSupportedMIMETypes();
    return set.get().get();
}

static Lock qlPreviewConverterDictionaryLock;

static NSMutableDictionary *QLPreviewConverterDictionary() WTF_REQUIRES_LOCK(qlPreviewConverterDictionaryLock)
{
    static NSMutableDictionary *dictionary = [[NSMutableDictionary alloc] init];
    return dictionary;
}

static NSMutableDictionary *QLContentDictionary()
{
    static NSMutableDictionary *contentDictionary = [[NSMutableDictionary alloc] init];
    return contentDictionary;
}

void removeQLPreviewConverterForURL(NSURL *url)
{
    Locker locker { qlPreviewConverterDictionaryLock };
    [QLPreviewConverterDictionary() removeObjectForKey:url];
    [QLContentDictionary() removeObjectForKey:url];
}

static void addQLPreviewConverterWithFileForURL(NSURL *url, id converter, NSString *fileName)
{
    ASSERT(url);
    ASSERT(converter);
    Locker locker { qlPreviewConverterDictionaryLock };
    [QLPreviewConverterDictionary() setObject:converter forKey:url];
    [QLContentDictionary() setObject:(fileName ? fileName : @"") forKey:url];
}

RetainPtr<NSURLRequest> registerQLPreviewConverterIfNeeded(NSURL *url, NSString *mimeType, NSData *data)
{
    RetainPtr<NSString> updatedMIMEType = adoptNS(PAL::softLink_QuickLook_QLTypeCopyBestMimeTypeForURLAndMimeType(url, mimeType));

    if ([QLPreviewGetSupportedMIMETypesSet() containsObject:updatedMIMEType.get()]) {
        RetainPtr<NSString> uti = adoptNS(PAL::softLink_QuickLook_QLTypeCopyUTIForURLAndMimeType(url, updatedMIMEType.get()));

        auto converter = adoptNS([PAL::allocQLPreviewConverterInstance() initWithData:data name:nil uti:uti.get() options:nil]);
        ResourceRequest previewRequest = [converter previewRequest];

        // We use [request URL] here instead of url since it will be
        // the URL that the WebDataSource will see during -dealloc.
        addQLPreviewConverterWithFileForURL(previewRequest.url(), converter.get(), nil);

        return previewRequest.nsURLRequest(HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
    }

    return nil;
}

bool isQuickLookPreviewURL(const URL& url)
{
    return url.protocolIs(QLPreviewProtocol);
}

static NSDictionary *temporaryFileAttributes()
{
    static NeverDestroyed<RetainPtr<NSDictionary>> attributes = @{
        NSFileOwnerAccountName : NSUserName(),
        NSFilePosixPermissions : [NSNumber numberWithInteger:(WEB_UREAD | WEB_UWRITE)],
    };
    return attributes.get().get();
}

static NSDictionary *temporaryDirectoryAttributes()
{
    static NeverDestroyed<RetainPtr<NSDictionary>> attributes = @{
        NSFileOwnerAccountName : NSUserName(),
        NSFilePosixPermissions : [NSNumber numberWithInteger:(WEB_UREAD | WEB_UWRITE | WEB_UEXEC)],
        NSFileProtectionKey : NSFileProtectionCompleteUnlessOpen,
    };
    return attributes.get().get();
}

NSString *createTemporaryFileForQuickLook(NSString *fileName)
{
    NSString *downloadDirectory = FileSystem::createTemporaryDirectory(@"QuickLookContent");
    if (!downloadDirectory)
        return nil;

    NSFileManager *fileManager = [NSFileManager defaultManager];

    NSError *error;
    if (![fileManager setAttributes:temporaryDirectoryAttributes() ofItemAtPath:downloadDirectory error:&error]) {
        LOG_ERROR("Failed to set attribute NSFileProtectionCompleteUnlessOpen on directory %@ with error: %@.", downloadDirectory, error.localizedDescription);
        return nil;
    }

    NSString *contentPath = [downloadDirectory stringByAppendingPathComponent:fileName.lastPathComponent];
    if (![fileManager _web_createFileAtPath:contentPath contents:nil attributes:temporaryFileAttributes()]) {
        LOG_ERROR("Failed to create QuickLook temporary file at path %@.", contentPath);
        return nil;
    }

    return contentPath;
}

} // namespace WebCore

#endif // USE(QUICK_LOOK)
