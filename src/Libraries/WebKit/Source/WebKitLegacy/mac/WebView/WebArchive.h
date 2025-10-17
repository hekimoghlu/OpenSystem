/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#import <WebKitLegacy/WebKitAvailability.h>

@class WebArchivePrivate;
@class WebResource;

/*!
    @const WebArchivePboardType
    @abstract The pasteboard type constant used when adding or accessing a WebArchive on the pasteboard.
*/
extern NSString *WebArchivePboardType WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @class WebArchive
    @discussion WebArchive represents a main resource as well as all the subresources and subframes associated with the main resource.
    The main resource can be an entire web page, a portion of a web page, or some other kind of data such as an image.
    This class can be used for saving standalone web pages, representing portions of a web page on the pasteboard, or any other
    application where one class is needed to represent rich web content. 
*/
WEBKIT_CLASS_DEPRECATED_MAC(10_3, 10_14)
@interface WebArchive : NSObject <NSCoding, NSCopying>
{
@package
    WebArchivePrivate *_private;
}

/*!
    @method initWithMainResource:subresources:subframeArchives:
    @abstract The initializer for WebArchive.
    @param mainResource The main resource of the archive.
    @param subresources The subresources of the archive (can be nil).
    @param subframeArchives The archives representing the subframes of the archive (can be nil).
    @result An initialized WebArchive.
*/
- (instancetype)initWithMainResource:(WebResource *)mainResource subresources:(NSArray *)subresources subframeArchives:(NSArray *)subframeArchives;

/*!
    @method initWithData:
    @abstract The initializer for creating a WebArchive from data.
    @param data The data representing the archive. This can be obtained using WebArchive's data method.
    @result An initialized WebArchive.
*/
- (instancetype)initWithData:(NSData *)data;

/*!
    @property mainResource
    @abstract The main resource of the archive.
*/
@property (nonatomic, readonly, strong) WebResource *mainResource;

/*!
    @property subresources
    @abstract The subresource of the archive (can be nil).
*/
@property (nonatomic, readonly, copy) NSArray *subresources;

/*!
    @property subframeArchives
    @abstract The archives representing the subframes of the archive (can be nil).
*/
@property (nonatomic, readonly, copy) NSArray *subframeArchives;

/*!
    @property data
    @abstract The data representation of the archive.
    @discussion The data returned by this method can be used to save a web archive to a file or to place a web archive on the pasteboard
    using WebArchivePboardType. To create a WebArchive using the returned data, call initWithData:.
*/
@property (nonatomic, readonly, copy) NSData *data;

@end
