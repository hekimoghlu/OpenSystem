/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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

#import "MBCBoardCommon.h"

NS_ASSUME_NONNULL_BEGIN

@class MBCBoardMTLView;

const float kDegrees2Radians = M_PI / 180.0f;

@interface MBCMetalCamera : NSObject

/*!
 @abstract Camera position in world space
 */
@property (nonatomic, readonly) vector_float3 position;

/*!
 @abstract The horizontal angle of the camera about the vertical (Y) axis of the board. Updated as drag the board to change viewing angle.
 */
@property (nonatomic) float azimuth;

/*!
 @abstract The vertical angle of the camera relative to the horizontal plane of the board in Degrees.
 */
@property (nonatomic) float elevation;

/*
 @abstract The viewport origin and size in pixels (x, y, width, height)
*/
@property (nonatomic, readonly) vector_float4 viewport;

/*!
 @abstract The current projection matrix based on size of the MTKView.
*/
@property (nonatomic, readonly) matrix_float4x4 projectionMatrix;

/*!
 @abstract The current view matrix based on position of camera.
*/
@property (nonatomic, readonly) matrix_float4x4 viewMatrix;

/*!
 @abstract Compute and store the view projection matrix when either of them change.
 Used for the conversion of positions from world to screen coordinates
 */
@property (nonatomic, readonly) matrix_float4x4 viewProjectionMatrix;

/*!
 @abstract The current view matrix for reflection map generation based on position of camera positioned below board.
*/
@property (nonatomic, readonly) matrix_float4x4 reflectionViewMatrix;

/*!
 @abstract Compute and store the view projection matrix when either projection or reflection view matrix changes.
 Used for the conversion of positions from world to screen coordinates for computing reflection map
*/
@property (nonatomic, readonly) matrix_float4x4 reflectionViewProjectionMatrix;

/*!
 @abstract initWithSize:
 @param size Current MTKView size in pixels.
 @discussion Creates a new camera for updating the view and projection matrices for Metal rendering.
*/
- (instancetype)initWithSize:(vector_float2)size;

/*!
 @abstract updateSize:
 @param size New size of MTKView in pixels.
 @discussion Updates the size of the view, which the camera will used to generate new projection matrix.
*/
- (void)updateSize:(vector_float2)size;

/*!
 @abstract The following three methods encapsulate conversion of position coordinates between world and screen coordinate spaces.
 */

/*!
 @abstract projectPositionFromModelToScreen:
 @param inPosition World position
 @discussion Converts world position in 3D to point on screen
 */
- (NSPoint)projectPositionFromModelToScreen:(MBCPosition)inPosition;

/*!
 @abstract unProjectPositionFromScreenToModel:fromView:
 @param position The screen position to convert to world coordinates
 @param view The instance of the MTLView for renderer
 @discussion Unprojects a screen position from screen to world coordinates.
 */
- (MBCPosition)unProjectPositionFromScreenToModel:(vector_float2)position fromView:(MBCBoardMTLView *)view;

/*!
 @abstract unProjectPositionFromScreenToModel:fromView:knownY:
 @param position The screen position to convert to world coordinates
 @param knownY The known y position in in world space
 @discussion Unprojects a screen position from screen to world coordinates for a constant y world value.
 */
- (MBCPosition)unProjectPositionFromScreenToModel:(vector_float2)position knownY:(float)knownY;

@end

NS_ASSUME_NONNULL_END
