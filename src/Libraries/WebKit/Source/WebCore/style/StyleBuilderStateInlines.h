/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

#include "RenderStyleSetters.h"
#include "StyleBuilderState.h"

namespace WebCore {
namespace Style {

inline const FontCascadeDescription& BuilderState::fontDescription() { return m_style.fontDescription(); }
inline const FontCascadeDescription& BuilderState::parentFontDescription() { return parentStyle().fontDescription(); }
inline void BuilderState::setUsedZoom(float zoom) { m_fontDirty |= m_style.setUsedZoom(zoom); }
inline void BuilderState::setFontDescription(FontCascadeDescription&& description) { m_fontDirty |= m_style.setFontDescriptionWithoutUpdate(WTFMove(description)); }
inline void BuilderState::setTextOrientation(TextOrientation orientation) { m_fontDirty |= m_style.setTextOrientation(orientation); }
inline void BuilderState::setWritingMode(StyleWritingMode mode) { m_fontDirty |= m_style.setWritingMode(mode); }
inline void BuilderState::setZoom(float zoom) { m_fontDirty |= m_style.setZoom(zoom); }

}
}
