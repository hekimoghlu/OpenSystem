/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
#pragma once

#if USE(TEXTURE_MAPPER)

#include "TextureMapperGLHeaders.h"
#include "TransformationMatrix.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

#define TEXMAP_ATTRIBUTE_VARIABLES(macro) \
    macro(vertex) \

#define TEXMAP_UNIFORM_VARIABLES(macro) \
    macro(modelViewMatrix) \
    macro(projectionMatrix) \
    macro(textureSpaceMatrix) \
    macro(textureColorSpaceMatrix) \
    macro(opacity) \
    macro(color) \
    macro(yuvToRgb) \
    macro(filterAmount) \
    macro(texelSize) \
    macro(gaussianKernel) \
    macro(gaussianKernelOffset) \
    macro(gaussianKernelHalfSize) \
    macro(blurDirection) \
    macro(roundedRectNumber) \
    macro(roundedRect) \
    macro(roundedRectInverseTransformMatrix)

#define TEXMAP_SAMPLER_VARIABLES(macro)           \
    macro(sampler)                                \
    macro(samplerY)                               \
    macro(samplerU)                               \
    macro(samplerV)                               \
    macro(samplerA)                               \
    macro(mask)                                   \
    macro(contentTexture)                         \
    macro(externalOESTexture)

#define TEXMAP_VARIABLES(macro) \
    TEXMAP_ATTRIBUTE_VARIABLES(macro) \
    TEXMAP_UNIFORM_VARIABLES(macro) \
    TEXMAP_SAMPLER_VARIABLES(macro) \

#define TEXMAP_DECLARE_VARIABLE(Accessor, Name, Type) \
    GLuint Accessor##Location() { \
        return getLocation(VariableID::Accessor, Name, Type); \
    }

#define TEXMAP_DECLARE_UNIFORM(Accessor) TEXMAP_DECLARE_VARIABLE(Accessor, "u_"#Accessor""_s, UniformVariable)
#define TEXMAP_DECLARE_ATTRIBUTE(Accessor) TEXMAP_DECLARE_VARIABLE(Accessor, "a_"#Accessor""_s, AttribVariable)
#define TEXMAP_DECLARE_SAMPLER(Accessor) TEXMAP_DECLARE_VARIABLE(Accessor, "s_"#Accessor""_s, UniformVariable)

#define TEXMAP_DECLARE_VARIABLE_ENUM(name) name,

class TextureMapperShaderProgram : public RefCounted<TextureMapperShaderProgram> {
public:
    enum Option {
        TextureRGB       = 1L << 0,
        SolidColor       = 1L << 2,
        Opacity          = 1L << 3,
        Antialiasing     = 1L << 5,
        GrayscaleFilter  = 1L << 6,
        SepiaFilter      = 1L << 7,
        SaturateFilter   = 1L << 8,
        HueRotateFilter  = 1L << 9,
        BrightnessFilter = 1L << 10,
        ContrastFilter   = 1L << 11,
        InvertFilter     = 1L << 12,
        OpacityFilter    = 1L << 13,
        BlurFilter       = 1L << 14,
        AlphaBlur        = 1L << 15,
        ContentTexture   = 1L << 16,
        ManualRepeat     = 1L << 17,
        TextureYUV       = 1L << 18,
        TextureNV12      = 1L << 19,
        TextureNV21      = 1L << 20,
        TexturePackedYUV = 1L << 21,
        TextureExternalOES = 1L << 22,
        RoundedRectClip  = 1L << 23,
        Premultiply      = 1L << 24,
        TextureYUVA      = 1L << 25,
        TextureCopy      = 1L << 26,
        AlphaToShadow    = 1L << 27,
    };

    enum class VariableID {
        TEXMAP_VARIABLES(TEXMAP_DECLARE_VARIABLE_ENUM)
    };

    using Options = OptionSet<Option>;

    static Ref<TextureMapperShaderProgram> create(Options);
    virtual ~TextureMapperShaderProgram();

    GLuint programID() const { return m_id; }


    TEXMAP_ATTRIBUTE_VARIABLES(TEXMAP_DECLARE_ATTRIBUTE)
    TEXMAP_UNIFORM_VARIABLES(TEXMAP_DECLARE_UNIFORM)
    TEXMAP_SAMPLER_VARIABLES(TEXMAP_DECLARE_SAMPLER)

    void setMatrix(GLuint, const TransformationMatrix&);

private:
    TextureMapperShaderProgram(const String& vertexShaderSource, const String& fragmentShaderSource);

    GLuint m_vertexShader;
    GLuint m_fragmentShader;

    enum VariableType { UniformVariable, AttribVariable };
    GLuint getLocation(VariableID, ASCIILiteral, VariableType);

    GLuint m_id;
    UncheckedKeyHashMap<VariableID, GLuint, IntHash<VariableID>, WTF::StrongEnumHashTraits<VariableID>> m_variables;
};

} // namespace WebCore

#endif // USE(TEXTURE_MAPPER)
