/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#import <simd/simd.h>
#import "MBCBoardEnums.h"

NS_ASSUME_NONNULL_BEGIN

@protocol MTLDevice;
@class MBCArrowInstance;
@class MBCBoardMTLView;
@class MBCBoardDecalInstance;
@class MBCMetalCamera;
@class MBCMetalMaterials;
@class MBCPieceInstance;

@interface MBCMetalRenderer : NSObject

/*!
 @abstract The default MTLDevice for rendering Chess games
 */
@property (nonatomic, class, strong, readonly) id<MTLDevice> defaultMTLDevice;

/*!
 @abstract The camera for Metal scene
 */
@property (nonatomic, strong, readonly) MBCMetalCamera *camera;

/*!
 @abstract initWithDevice:mtkView:
 @param device The default Metal device for rendering
 @param mtkView The MBCBoardMTLView (MTKView) used to render Metal content for chess
 @discussion Creates a new renderer used to render all Metal content for chess
*/
- (instancetype)initWithDevice:(id<MTLDevice>)device mtkView:(MBCBoardMTLView *)mtkView;

/*!
 @abstract drawableSizeWillChange:
 @param size Current MTKView size in pixels.
 @discussion Will adjust the size of drawable texture content based on new screen size
*/
- (void)drawableSizeWillChange:(CGSize)size;

/*!
 @abstract drawSceneToView
 @discussion Will execute the render commands needed to render content for current frame
*/
- (void)drawSceneToView;

/*!
 @abstract readPixel:
 @param position Current position of mouse in pixels
 @discussion Will use the main depth attachment to sample the distance from camera.
*/
- (float)readPixel:(vector_float2)position;

/*!
 @abstract cameraDidRotateAboutYAxis
 @discussion Called after the value of the camera's azimuth angle is updated.
 */
- (void)cameraDidRotateAboutYAxis;

/*!
 @abstract setWhitePieceInstances:blackPieceInstances:
 @param whiteInstances Multidimensional array where each element is an array of MBCPieceInstances
 @param blackInstances Multidimensional array where each element is an array of MBCPieceInstances
 @param transparentInstance Nullable instance that uses transparency
 @discussion Updates the renderable chess piece instances with data for white and black pieces. Each parameter is a
 multidimensional array where each element is an array of MBCPieceInstance for a given type of piece.  
*/
- (void)setWhitePieceInstances:(NSArray *)whiteInstances 
           blackPieceInstances:(NSArray *)blackInstances
           transparentInstance:(MBCPieceInstance * _Nullable)transparentInstance;


/*!
 @abstract setHintMoveInstance:lastMoveInstance:
 @param hintInstance Data needed to render the hint arrow instance
 @param lastMoveInstance Data needed to render the last move arrow instance
 @discussion Updates the renderable board arrows to indicate hint move and/or last move
*/
- (void)setHintMoveInstance:(MBCArrowInstance *_Nullable)hintInstance
           lastMoveInstance:(MBCArrowInstance *_Nullable)lastMoveInstance;

/*!
 @abstract setLabelInstances:
 @param labelInstances Array of MBCBoardDecalInstance objects for current frame in hand piece counts
 @discussion The Crazy House game variant stores captured pieces on side of board. Will draw labels
 for each piece count if a piece's count is > 1 in hand.
 */
- (void)setLabelInstances:(NSArray *)labelInstances;

/*!
 @abstract setPieceSelectionInstance:
 @param instance Instance encapsulating data needed to render the piece selection
 @discussion Called to update the position and visibility data for the piece selection graphic displayed under selected piece.
 */
- (void)setPieceSelectionInstance:(MBCBoardDecalInstance * _Nullable)instance;

/*!
 @abstract loadMaterialsForNewStyle:
 @param newStyle The name of the style to use for the board and pieces
 @discussion Loads the MBCDrawStyle instances that define the materials used to render the board and pieces in the scene.
*/
- (void)loadMaterialsForNewStyle:(NSString *)newStyle;

@end

NS_ASSUME_NONNULL_END
