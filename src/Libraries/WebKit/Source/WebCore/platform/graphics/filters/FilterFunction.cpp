/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "config.h"
#include "FilterFunction.h"

#include "ImageBuffer.h"
#include <wtf/SortedArrayMap.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

FilterFunction::FilterFunction(Type filterType, std::optional<RenderingResourceIdentifier> renderingResourceIdentifier)
    : RenderingResource(renderingResourceIdentifier)
    , m_filterType(filterType)
{
}

AtomString FilterFunction::filterName(Type filterType)
{
    static constexpr std::pair<FilterFunction::Type, ASCIILiteral> namesArray[] = {
        { FilterFunction::Type::CSSFilter,           "CSSFilter"_s           },
        { FilterFunction::Type::SVGFilter,           "SVGFilter"_s           },
        
        { FilterFunction::Type::FEBlend,             "FEBlend"_s             },
        { FilterFunction::Type::FEColorMatrix,       "FEColorMatrix"_s       },
        { FilterFunction::Type::FEComponentTransfer, "FEComponentTransfer"_s },
        { FilterFunction::Type::FEComposite,         "FEComposite"_s         },
        { FilterFunction::Type::FEConvolveMatrix,    "FEConvolveMatrix"_s    },
        { FilterFunction::Type::FEDiffuseLighting,   "FEDiffuseLighting"_s   },
        { FilterFunction::Type::FEDisplacementMap,   "FEDisplacementMap"_s   },
        { FilterFunction::Type::FEDropShadow,        "FEDropShadow"_s        },
        { FilterFunction::Type::FEFlood,             "FEFlood"_s             },
        { FilterFunction::Type::FEGaussianBlur,      "FEGaussianBlur"_s      },
        { FilterFunction::Type::FEImage,             "FEImage"_s             },
        { FilterFunction::Type::FEMerge,             "FEMerge"_s             },
        { FilterFunction::Type::FEMorphology,        "FEMorphology"_s        },
        { FilterFunction::Type::FEOffset,            "FEOffset"_s            },
        { FilterFunction::Type::FESpecularLighting,  "FESpecularLighting"_s  },
        { FilterFunction::Type::FETile,              "FETile"_s              },
        { FilterFunction::Type::FETurbulence,        "FETurbulence"_s        },

        { FilterFunction::Type::SourceAlpha,         "SourceAlpha"_s         },
        { FilterFunction::Type::SourceGraphic,       "SourceGraphic"_s       }
    };

    static constexpr SortedArrayMap namesMap { namesArray };
    
    ASSERT(namesMap.tryGet(filterType));
    return namesMap.get(filterType, ""_s);
}

TextStream& operator<<(TextStream& ts, const FilterFunction& filterFunction)
{
    return filterFunction.externalRepresentation(ts, FilterRepresentation::Debugging);
}

} // namespace WebCore
