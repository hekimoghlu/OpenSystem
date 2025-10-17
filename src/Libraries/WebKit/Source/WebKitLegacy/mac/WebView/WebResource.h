/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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

@class WebResourcePrivate;


/*!
    @class WebResource
    @discussion A WebResource represents a fully downloaded URL. 
    It includes the data of the resource as well as the metadata associated with the resource.
*/
@interface WebResource : NSObject <NSCoding, NSCopying>
{
@package
    WebResourcePrivate *_private;
}

/*!
    @method initWithData:URL:MIMEType:textEncodingName:frameName
    @abstract The initializer for WebResource.
    @param data The data of the resource.
    @param URL The URL of the resource.
    @param MIMEType The MIME type of the resource.
    @param textEncodingName The text encoding name of the resource (can be nil).
    @param frameName The frame name of the resource if the resource represents the contents of an entire HTML frame (can be nil).
    @result An initialized WebResource.
*/
- (instancetype)initWithData:(NSData *)data URL:(NSURL *)URL MIMEType:(NSString *)MIMEType textEncodingName:(NSString *)textEncodingName frameName:(NSString *)frameName;

/*!
    @property data
    @abstract The data of the resource.
*/
@property (nonatomic, readonly, copy) NSData *data;

/*!
    @property URL
    @abstract The URL of the resource.
*/
@property (nonatomic, readonly, strong) NSURL *URL;

/*!
    @property MIMEType
    @abstract The MIME type of the resource.
*/
@property (nonatomic, readonly, copy) NSString *MIMEType;

/*!
    @property textEncodingName
    @abstract The text encoding name of the resource (can be nil).
*/
@property (nonatomic, readonly, copy) NSString *textEncodingName;

/*!
    @property frameName
    @abstract The frame name of the resource if the resource represents the contents of an entire HTML frame (can be nil).
*/
@property (nonatomic, readonly, copy) NSString *frameName;

@end
