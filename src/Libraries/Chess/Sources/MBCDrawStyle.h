/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#import "MBCShaderTypes.h"

@protocol MTLTexture;

@interface MBCDrawStyle : NSObject {
@public
    /*!
     @abstract The diffuse color component
    */
    float fDiffuse;
    
    /*!
     @abstract The specular color component
    */
    float fSpecular;
    
    /*!
     @abstract The shininess to use to render model
    */
    float fShininess;
    
    /*!
     @abstract The alpha color component to control opacity of model
    */
    float fAlpha;
    
    /*!
     @abstract PBR rendering settings to use when material maps are not available for these attributes.
     */
    MBCSimpleMaterial fMaterial;

@private
    /*!
     @abstract OpenGL texture index
    */
    uint32_t fTexture;
    
    /*!
     @abstract Reference to the base color texture map
    */
    id<MTLTexture> fBaseColorTexture;
}

/*!
 @abstract Texture used for the model's base color
*/
@property (nonatomic, strong) id<MTLTexture> baseColorTexture;

/*!
 @abstract Texture used for the model's normal map
*/
@property (nonatomic, strong) id<MTLTexture> normalMapTexture;

/*!
 @abstract Texture used for the model's roughness and ambient occlusion PBR parameters. Red channel is
 the roughness value and Green is ambient occlusion. Green is unused for now.
*/
@property (nonatomic, strong) id<MTLTexture> roughnessAmbientOcclusionTexture;

/*!
 @abstract init:
 @discussion Default initializer for MBCDrawStyle
*/
- (instancetype)init;

/*!
 @abstract initWithTexture:
 @param tex OpenGL texture id for texture
 @discussion This is the initializer used when creating MBCDrawStyle for OpenGL rendering.
*/
- (instancetype)initWithTexture:(uint32_t)tex;

/*!
 @abstract unloadTexture:
 @discussion Unloads the OpenGL texture, not needed for Metal Rendering
*/
- (void)unloadTexture;

/*!
 @abstract startStyle:
 @param alpha The alpha channel value to use for material.
 @discussion This is used by OpenGL to set the material parameters for currently drawn 3D asset.
*/
- (void)startStyle:(float)alpha;

/*!
 @abstract materialForPBR
 @discussion This method will update the base color texture for the draw style.
*/
- (MBCSimpleMaterial)materialForPBR;

/*!
 @abstract updateMTLTexture:
 @param texture the MTLTexture to use for base color for this style
 @discussion This method will update the base color texture for the draw style.
*/
- (void)updateMTLTexture:(id<MTLTexture>)texture;

@end
