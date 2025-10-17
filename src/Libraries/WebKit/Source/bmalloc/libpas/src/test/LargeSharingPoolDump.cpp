/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "LargeSharingPoolDump.h"

#if TLC

#include "pas_page_sharing_pool.h"

using namespace std;

namespace {

bool forEachAdapter(pas_large_sharing_node* node,
                    void* arg)
{
    function<bool(pas_large_sharing_node*)>* visitor =
        static_cast<function<bool(pas_large_sharing_node*)>*>(arg);
    
    return (*visitor)(node);
}

} // anonymous namespace

void forEachLargeSharingPoolNode(function<bool(pas_large_sharing_node*)> visitor)
{
    pas_large_sharing_pool_for_each(forEachAdapter, &visitor, pas_lock_is_not_held);
}

vector<pas_large_sharing_node*> largeSharingPoolAsVector()
{
    vector<pas_large_sharing_node*> result;
    forEachLargeSharingPoolNode([&] (pas_large_sharing_node* node) -> bool {
        result.push_back(node);
        return true;
    });
    return result;
}

void dumpLargeSharingPool()
{
    cout << "Large sharing pool:\n";
    cout.flush();
    forEachLargeSharingPoolNode(
        [&] (pas_large_sharing_node* node) -> bool {
            cout << "    " << reinterpret_cast<void*>(node->range.begin)
                 << "..." << reinterpret_cast<void*>(node->range.end) << ": use_epoch = "
                 << node->use_epoch << ", num_live_bytes = " << node->num_live_bytes
                 << ", is_committed = " << !!node->is_committed << "\n";
            cout.flush();
            return true;
        });
    cout << "Physical balance: " << pas_physical_page_sharing_pool_balance << "\n";
    cout.flush();
}

#endif // TLC
