/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#include "Model.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<Model> Model::create(Ref<SharedBuffer>&& data, String mimeType, URL url)
{
    return adoptRef(*new Model(WTFMove(data), WTFMove(mimeType), WTFMove(url)));
}

Model::Model(Ref<SharedBuffer>&& data, String mimeType, URL url)
    : m_data(WTFMove(data))
    , m_mimeType(WTFMove(mimeType))
    , m_url(WTFMove(url))
{
}

Model::~Model() = default;

TextStream& operator<<(TextStream& ts, const Model& model)
{
    TextStream::GroupScope groupScope(ts);

    ts.dumpProperty("data-size", model.data()->size());
    ts.dumpProperty("mime-type", model.mimeType());
    ts.dumpProperty("url", model.url());

    return ts;
}

}
