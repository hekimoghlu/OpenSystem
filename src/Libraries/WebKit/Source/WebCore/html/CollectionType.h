/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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

namespace WebCore {

enum class CollectionType : uint8_t {
    // Unnamed HTMLCollection types cached in the document.
    DocImages,    // all <img> elements in the document
    DocEmbeds,    // all <embed> elements
    DocEmpty,     // always empty (for document.applets)
    DocForms,     // all <form> elements
    DocLinks,     // all <a> _and_ <area> elements with a value for href
    DocAnchors,   // all <a> elements with a value for name
    DocScripts,   // all <script> elements
    DocAll,       // "all" elements (IE)

    // Named collection types cached in the document.
    WindowNamedItems,
    DocumentNamedItems,

    DocumentAllNamedItems, // Sub-collection returned by the "all" collection when there are multiple items with the same name

    // Unnamed HTMLCollection types cached in elements.
    NodeChildren, // first-level children (IE)
    TableTBodies, // all <tbody> elements in this table
    TSectionRows, // all row elements in this table section
    TableRows,
    TRCells,      // all cells in this row
    SelectOptions,
    SelectedOptions,
    DataListOptions,
    MapAreas,
    FormControls,
    FieldSetElements,
    ByClass,
    ByTag,
    ByHTMLTag,
    AllDescendants
};

enum class CollectionTraversalType : uint8_t { Descendants, ChildrenOnly, CustomForwardOnly };
template<CollectionType collectionType>
struct CollectionTypeTraits {
    static const CollectionTraversalType traversalType = CollectionTraversalType::Descendants;
};

template<>
struct CollectionTypeTraits<CollectionType::NodeChildren> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::ChildrenOnly;
};

template<>
struct CollectionTypeTraits<CollectionType::TRCells> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::ChildrenOnly;
};

template<>
struct CollectionTypeTraits<CollectionType::TSectionRows> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::ChildrenOnly;
};

template<>
struct CollectionTypeTraits<CollectionType::TableTBodies> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::ChildrenOnly;
};

template<>
struct CollectionTypeTraits<CollectionType::TableRows> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::CustomForwardOnly;
};

template<>
struct CollectionTypeTraits<CollectionType::FormControls> {
    static const CollectionTraversalType traversalType = CollectionTraversalType::CustomForwardOnly;
};

} // namespace WebCore
