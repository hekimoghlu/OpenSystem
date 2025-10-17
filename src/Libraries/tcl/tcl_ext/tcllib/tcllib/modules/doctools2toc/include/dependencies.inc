/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 8, 2024.
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

[comment {
         ASCII diagram of the dependencies between the doctools v2 toc packages
         ======================================================================
}][example {
                                    ~~~~~~~~~~~ doctools::toc ~~~~~~~~~~~
                                   ~~                   |               ~~
                doctools::toc::export ~~~~~~~~~~~~~~~~~ | ~~~~~~~~~~~~~ doctools::toc::import
                        |                               |                       |
        +---------------+-------------------------+     |    +------------------+---------------+-----------------------+---------------+
        |               |                         |     |    |                  |               |                       |               |
doctools::config        =                         |     |    |                  =       doctools::include       doctools::config doctools::paths
                        |                         |     |    |                  |
                doctools::toc::export::<*>        |     |    |          doctools::toc::import::<*>
                        doctoc                    |     |    |                  doctoc, json
                        json                      |     |    |                  |           \\
                        html                      |     |    |          doctools::toc::parse \\
                        nroff                     |     |    |                  |             \\
                        wiki                      |     |    |  +---------------+              json
                        text                      |     |    |  |               |
                                                doctools::toc::structure        |
                                                                                |
                                                                        +-------+---------------+
                                                                        |                       |
          doctools::html  doctools::html::cssdefaults           doctools::tcl::parse    doctools::msgcat        
                |                                                                               |
          doctools::text  doctools::nroff::man_macros                                           =
                                                                                                |
                                                                                        doctools::msgcat::toc::<*>
                                                                                                c, en, de, fr
                                                                                                (fr == en for now)
        ~~      Interoperable objects, without actual package dependencies
        --      Package dependency, higher requires lower package
        =       Dynamic dependency through plugin system
        <*>     Multiple packages following the given form of naming.

}][comment {
        yaml                    export, import
        tmml                    export, import
        reStructured Text       export
        latex                   export

        list, desc - old, not needed under new system, replaced by the nested-list serialization
        null - old, not needed, deserialize doctoc alone provides validation of input.
}]
