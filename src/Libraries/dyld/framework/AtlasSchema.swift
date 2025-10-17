/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
// TODO: Do some nice ASCII artwork for the schema, for now the validation code will serve as documentation

protocol AtlasSchemaValidator {
    // validate the schema of a bplist object corresponding to a give class, recusrively
    // All validation failures throw, so we can (eventually) encode richer failure information
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError)
}


extension Snapshot.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        try bplist.validate(key:"proc",     as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"plat",     as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"time",     as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"stat",     as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"init",     as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"flags",    as:Int64.self,  throwAs:AtlasError.self,                optional:true)
        try bplist.validate(key:"suid",     as:Data.self,   throwAs:AtlasError.self,    count:16,   optional:true)
        try bplist.validateArray(key:"imgs", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try Image.Impl.validate(bplist:bplist)
        }
        try bplist.validateArray(key:"aots", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self, optional:true) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try AOTImage.Impl.validate(bplist:bplist)
        }
        if let sharedCacheProcessRecord = bplist["envp", as:BPList.UnsafeRawDictionary.self] {
            try Environment.Impl.validate(bplist:sharedCacheProcessRecord)
        }
        if let sharedCacheProcessRecord = bplist["dsc1", as:BPList.UnsafeRawDictionary.self] {
            try SharedCache.ProcessRecord.validate(bplist:sharedCacheProcessRecord)
        }
    }
}

extension Environment.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        try bplist.validate(key:"root",    as:String.self,  throwAs:AtlasError.self,                optional:true)
    }
}

extension SharedCache.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        try bplist.validate(key:"uuid", as:Data.self,   throwAs:AtlasError.self,    count:16)
        try bplist.validate(key:"padr", as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"psze", as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"size", as:Int64.self,  throwAs:AtlasError.self)
        try bplist.validate(key:"snme", as:String.self, throwAs:AtlasError.self,                optional:true)
        try bplist.validate(key:"suid", as:Data.self,   throwAs:AtlasError.self,    count:16,   optional:true)
        try bplist.validateArray(key:"imgs", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try Image.Impl.validate(bplist:bplist)
        }
        try bplist.validateArray(key:"aots", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self, optional:true) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try AOTImage.Impl.validate(bplist:bplist)
        }
    }
}

extension SharedCache.ProcessRecord: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        do throws(BPListError) {
            try bplist.validate(key:"uuid", as:Data.self,   count:16)
            try bplist.validate(key:"bitm", as:Data.self)
            try bplist.validate(key:"addr", as:Int64.self)
            try bplist.validate(key:"file", as:String.self)
            try bplist.validate(key:"aadr", as:Int64.self,              optional:true)
            try bplist.validate(key:"auid", as:Data.self,   count:16,   optional:true)
        } catch {
            throw .bplistError(error)
        }
    }
}

extension SubCache.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        try bplist.validate(key:"uuid", as:Data.self,   throwAs:AtlasError.self,    count:16)
        try bplist.validate(key:"name", as:String.self, throwAs:AtlasError.self)
        try bplist.validate(key:"size", as:Data.self,   throwAs:AtlasError.self)
        try bplist.validate(key:"fsze", as:Data.self,   throwAs:AtlasError.self)
        try bplist.validate(key:"voff", as:Data.self,   throwAs:AtlasError.self)
        try bplist.validate(key:"padr", as:Data.self,   throwAs:AtlasError.self)
        try bplist.validate(key:"suid", as:Data.self,   throwAs:AtlasError.self,    count:16,   optional:true)
        try bplist.validateArray(key:"maps", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try SubCache.Mapping.validate(bplist:bplist)
        }
    }
}

extension SubCache.Mapping: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        do {
            try bplist.validate(key:"padr", as:Int64.self)
            try bplist.validate(key:"size", as:Int64.self)
            try bplist.validate(key:"foff", as:Int64.self)
            try bplist.validate(key:"prot", as:Int64.self)
        } catch {
            throw .bplistError(error)
        }
    }
}

extension Image.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        var validAddress = true
        do throws(BPListError) {
            let hasPreferredAddress =   try bplist.validate(key:"padr", as:Int64.self,              optional:true)
            let hasAddress          =   try bplist.validate(key:"addr", as:Int64.self,              optional:true)
                                        try bplist.validate(key:"uuid", as:Data.self,   count:16,   optional:true)
                                        try bplist.validate(key:"name", as:String.self,             optional:true)
                                        try bplist.validate(key:"file", as:String.self,             optional:true)
            if hasPreferredAddress == nil, hasAddress == nil {
                // All shared cache images have a preferred load address, and all non-shared cache images have a load address (as well
                // as a preferred load address if they are not built to load at address zero). If neither are present the image record
                // is malformed
                validAddress = false
            }
        } catch {
            throw .bplistError(error)
        }
        guard validAddress == true else {
            throw .schemaValidationError
        }
        try bplist.validateArray(key:"segs", of:BPList.UnsafeRawDictionary.self, throwAs:AtlasError.self) { (bplist:BPList.UnsafeRawDictionary) throws(AtlasError) -> Void in
            return try Segment.Impl.validate(bplist:bplist)
        }
    }
}

extension Segment.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        do throws(BPListError) {
            try bplist.validate(key:"uuid", as:Data.self, count:16, optional:true)
            try bplist.validate(key:"name", as:String.self)
            try bplist.validate(key:"size", as:Int64.self)
            try bplist.validate(key:"fsze", as:Int64.self)
            try bplist.validate(key:"padr", as:Int64.self)
            try bplist.validate(key:"perm", as:Int64.self)
        } catch {
            throw .bplistError(error)
        }
    }
}

extension  AOTImage.Impl: AtlasSchemaValidator {
    static func validate(bplist: BPList.UnsafeRawDictionary) throws(AtlasError) {
        do throws(BPListError) {
            try bplist.validate(key:"xadr", as:Int64.self)
            try bplist.validate(key:"aadr", as:Int64.self)
            try bplist.validate(key:"asze", as:Int64.self)
            try bplist.validate(key:"ikey", as:Data.self)
        } catch {
            throw .bplistError(error)
        }
    }
}
