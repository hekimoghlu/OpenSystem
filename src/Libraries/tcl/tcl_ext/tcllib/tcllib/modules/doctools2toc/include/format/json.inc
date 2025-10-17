/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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


[section {JSON notation of tables of contents}]

The JSON format used for tables of contents is a direct translation of
the [sectref {ToC serialization format}], mapping Tcl dictionaries as
JSON objects and Tcl lists as JSON arrays.

For example, the Tcl serialization

[example {
doctools::toc {
    items {
        {reference {
	    desc {DocTools - Tables of Contents}
	     id introduction.man
	     label doctools::toc::introduction
	}}
	{division {
	     id processing.man
	     items {
	         {reference {
		     desc {doctoc serialization utilities}
		     id structure.man
		     label doctools::toc::structure
		 }}
		 {reference {
		     desc {Parsing text in doctoc format}
		     id parse.man
		     label doctools::toc::parse
		 }}
	     }
             label Processing
        }}
    }
    label {Table of Contents} 
    title TOC
}
}]

is equivalent to the JSON string

[example {
{
    "doctools::toc" : {
        "items" : [{
            "reference" : {
                "desc"  : "DocTools - Tables of Contents",
                "id"    : "introduction.man",
                "label" : "doctools::toc::introduction"
            }
        },{
            "division" : {
                "id"    : "processing.man",
                "items" : [{
                    "reference" : {
                        "desc"  : "doctoc serialization utilities",
                        "id"    : "structure.man",
                        "label" : "doctools::toc::structure"
                    }
                },{
                    "reference" : {
                        "desc"  : "Parsing text in doctoc format",
                        "id"    : "parse.man",
                        "label" : "doctools::toc::parse"
                    }
                }],
                "label" : "Processing"
            }
        }],
        "label" : "Table of Contents",
        "title" : "TOC"
    }
}
}]
