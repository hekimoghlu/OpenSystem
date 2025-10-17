/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#import "MBCDrawStyle.h"
#import "MBCShaderTypes.h"

#import <OpenGL/glu.h>
#import <Metal/Metal.h>

@implementation MBCDrawStyle

- (instancetype)init {
    self = [super init];
    if (self) {
        fTexture = 0;
    }
    return self;
}

- (instancetype)initWithTexture:(uint32_t)tex {
    self = [super init];
    if (self) {
        fTexture = tex;
        fDiffuse = 1.0f;
        fSpecular = 0.2f;
        fShininess = 5.0f;
        fAlpha = 1.0f;
        
        fMaterial.roughness = 0.f;
        fMaterial.ambientOcclusion = 1.f;
        fMaterial.metallic = 0.f;
    }
    return self;
}

- (void)unloadTexture {
    if (fTexture) {
        glDeleteTextures(1, &fTexture);
    }
}

- (void)startStyle:(float)alpha {
    GLfloat white_texture_color[4]     =
        {fDiffuse, fDiffuse, fDiffuse, fAlpha*alpha};
    GLfloat emission_color[4]         =
        {0.0f, 0.0f, 0.0f, fAlpha*alpha};
    GLfloat specular_color[4]         =
        {fSpecular, fSpecular, fSpecular, fAlpha*alpha};

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white_texture_color);
    glMaterialfv(GL_FRONT, GL_EMISSION, emission_color);
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular_color);
    glMaterialf(GL_FRONT, GL_SHININESS, fShininess);
    glBindTexture(GL_TEXTURE_2D, fTexture);
}

- (MBCSimpleMaterial)materialForPBR {
    return fMaterial;
}

- (void)updateMTLTexture:(id<MTLTexture>)texture {
    fBaseColorTexture = texture;
}

@end
