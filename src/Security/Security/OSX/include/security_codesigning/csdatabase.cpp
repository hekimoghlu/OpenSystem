/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
//
// csdb - system-supported Code Signing related database interfaces
//
#include "csdatabase.h"
#include "detachedrep.h"

namespace Security {
namespace CodeSigning {

using namespace SQLite;


//
// The one and only SignatureDatabase object.
// It auto-adapts to readonly vs. writable use.
//
ModuleNexus<SignatureDatabase> signatureDatabase;
ModuleNexus<SignatureDatabaseWriter> signatureDatabaseWriter;


//
// Default path to the signature database.
//
const char SignatureDatabase::defaultPath[] = "/private/var/db/DetachedSignatures";


//
// Creation commands to initialize the system database.
//
const char schema[] = "\
	create table if not exists code ( \n\
		id integer primary key on conflict replace autoincrement not null, \n\
		global integer null references global (id), \n\
		identifier text not null, \n\
		architecture integer, \n\
		identification blob not null unique on conflict replace, \n\
		signature blob not null, \n\
		created text default current_timestamp \n\
	); \n\
	create index if not exists identifier_index on code (identifier); \n\
	create index if not exists architecture_index on code (architecture); \n\
	create index if not exists id_index on code (identification); \n\
	\n\
	create table if not exists global ( \n\
		id integer primary key on conflict replace autoincrement not null, \n\
		sign_location text not null, \n\
		signature blob null \n\
	); \n\
	create index if not exists location_index on global (sign_location); \n\
";



//
// Open the database (creating it if necessary and possible).
// Note that this isn't creating the schema; we do that on first write.
//
SignatureDatabase::SignatureDatabase(const char *path, int flags)
	: SQLite::Database(path, flags, true)	// lenient open
{
}

SignatureDatabase::~SignatureDatabase()
{ /* virtual */ }


//
// Consult the database to find code by identification blob.
// Return the signature and (optional) global data blobs.
//
FilterRep *SignatureDatabase::findCode(DiskRep *rep)
{
	if (CFRef<CFDataRef> identification = rep->identification())
		if (!this->empty()) {
			SQLite::Statement query(*this,
				"select code.signature, global.signature from code, global \
				 where code.identification = ?1 and code.global = global.id;");
			query.bind(1) = identification.get();
			if (query.nextRow()) {
				CFRef<CFDataRef> sig = query[0].data();
				CFRef<CFDataRef> gsig = query[1].data();
				return new DetachedRep(sig, gsig, rep, "system");
			}
		}

	// no joy
	return NULL;
}


//
// Given a unified detached signature blob, store its data in the database.
// This writes exactly one Global record, plus one Code record per architecture
// (where non-architectural code is treated as single-architecture).
//
void SignatureDatabaseWriter::storeCode(const BlobCore *sig, const char *location)
{
	if (!this->isOpen())	// failed database open or creation
		MacOSError::throwMe(errSecCSDBAccess);
	Transaction xa(*this, Transaction::exclusive);	// lock out everyone
	if (this->empty())
		this->execute(schema);					// initialize schema
	if (const EmbeddedSignatureBlob *esig = EmbeddedSignatureBlob::specific(sig)) {	// architecture-less
		int64 globid = insertGlobal(location, NULL);
		insertCode(globid, 0, esig);
		xa.commit();
		return;
	} else if (const DetachedSignatureBlob *dsblob = DetachedSignatureBlob::specific(sig)) {
		int64 globid = insertGlobal(location, dsblob->find(0));
		unsigned count = dsblob->count();
		for (unsigned n = 0; n < count; n++)
			if (uint32_t arch = dsblob->type(n))
				insertCode(globid, arch, EmbeddedSignatureBlob::specific(dsblob->blob(n)));
		xa.commit();
		return;
	}
	
	MacOSError::throwMe(errSecCSSignatureInvalid);

}

int64 SignatureDatabaseWriter::insertGlobal(const char *location, const BlobCore *blob)
{
	Statement insert(*this, "insert into global (sign_location, signature) values (?1, ?2);");
	insert.bind(1) = location;
	if (blob)
		insert.bind(2).blob(blob, blob->length(), true);
	insert();
	return lastInsert();
}

void SignatureDatabaseWriter::insertCode(int64 globid, int arch, const EmbeddedSignatureBlob *sig)
{
	// retrieve binary identifier (was added by signer)
	const BlobWrapper *ident = BlobWrapper::specific(sig->find(cdIdentificationSlot));
	assert(ident);
	
	// extract CodeDirectory to get some information from it
	const CodeDirectory *cd = CodeDirectory::specific(sig->find(cdCodeDirectorySlot));
	assert(cd);

	// write the record
	Statement insert(*this,
		"insert into code (global, identifier, architecture, identification, signature) values (?1, ?2, ?3, ?4, ?5);");
	insert.bind(1) = globid;
	insert.bind(2) = cd->identifier();
	if (arch)
		insert.bind(3) = arch;
	insert.bind(4).blob(ident->data(), ident->length(), true);
	insert.bind(5).blob(sig, sig->length(), true);
	insert();
}



} // end namespace CodeSigning
} // end namespace Security
