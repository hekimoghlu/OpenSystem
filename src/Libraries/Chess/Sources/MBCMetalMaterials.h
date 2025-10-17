/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#import "MBCBoardEnums.h"

@class MBCDrawStyle;
@protocol MTLDevice;
@protocol MTLTexture;

NS_ASSUME_NONNULL_BEGIN

/*!
 @typedef MBCMaterialType
 @abstract MBCMaterialType is used to determine type of asset in which to load material
 @constant MBCMaterialTypeBoard Asset type is the board
 @constant MBCMaterialTypePiece Asset type is a piece
*/
typedef NS_ENUM(NSInteger, MBCMaterialType) {
    MBCMaterialTypeBoard = 0,
    MBCMaterialTypePiece
};

@interface MBCMetalMaterials : NSObject

/*!
 @abstract The cube map texture to use for additional diffuse lighting for scene.
 */
@property (nonatomic, strong, readonly) id<MTLTexture> irradianceMap;

/*!
 @abstract MTLTexture for the base color of the  ground plane
 */
@property (nonatomic, strong, readonly) id<MTLTexture> groundTexture;

/*!
 @abstract MTLTexture for the base color of the  edge notation labels
 */
@property (nonatomic, strong, readonly) id<MTLTexture> edgeNotationTexture;

/*!
 @abstract MTLTexture for the normal map  of the  edge notation labels
 */
@property (nonatomic, strong, readonly) id<MTLTexture> edgeNotationNormalTexture;

/*!
 @abstract MTLTexture for the base color of the  piece selection indicator
 */
@property (nonatomic, strong, readonly) id<MTLTexture> pieceSelectionTexture;

/*!
 @abstract MTLTexture for the base color hint arrow and the last move arrow
 */
@property (nonatomic, strong, readonly) id<MTLTexture> hintArrowTexture;
@property (nonatomic, strong, readonly) id<MTLTexture> lastMoveArrowTexture;

/*!
 @abstract shared
 @discussion Returns the shared instance of the MBCMetalMaterials. Singleton to allow for sharing resources across NSWindows.
 */
+ (instancetype)shared;

- (instancetype)init NS_UNAVAILABLE;

/*!
 @abstract loadMaterialsForRendererID:newStyle:
 @param rendererID The identifier string of renderer requesting material resources
 @param newStyle The name of the style to use for the board and chess pieces
 @discussion Loads the MBCDrawStyle instances that define the materials used to render the board and pieces in the scene.
*/
- (void)loadMaterialsForRendererID:(NSString *)rendererID newStyle:(NSString *)newStyle;

/*!
 @abstract drawStyleForPiece:style:
 @param piece The piece code for which to retrieve style
 @discussion Returns the draw style instance associated with piece parameter
*/
- (MBCDrawStyle *)materialForPiece:(MBCPiece)piece style:(NSString *)style;

/*!
 @abstract boardMaterialWithStyle:
 @discussion Returns the draw style instance associated with the board
*/
- (MBCDrawStyle *)boardMaterialWithStyle:(NSString *)style;

/*!
 @abstract releaseUsageForStyle:rendererID:
 @param style The name of the style being requested (Wood, Marble, Metal)
 @param rendererID The identifier string of renderer requesting material resources
 @discussion Called when a renderer stops using a style. If a style is no longer being used
 by any renderer then that style along with all its textures will be released
 */
- (void)releaseUsageForStyle:(NSString *)style rendererID:(NSString *)rendererID;

@end

NS_ASSUME_NONNULL_END
