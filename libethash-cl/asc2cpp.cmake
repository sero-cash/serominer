# Script to embed contents of a file as byte array in C/C++ header file(.h). The header file
# will contain a byte array and integer variable holding the size of the array.
# Parameters
#   SOURCE_FILE     - The path of source file whose contents will be embedded in the header file.
#   VARIABLE_NAME   - The name of the variable for the string literal.
#   HEADER_FILE     - The path of .cpp file.

include(CMakeParseArguments)

set(options APPEND NULL_TERMINATE)
set(oneValueArgs SOURCE_FILE VARIABLE_NAME HEADER_FILE)

# reads source file contents as hex string
file(READ ${BIN2H_SOURCE_FILE} asciiString)
file(WRITE ${BIN2H_HEADER_FILE} "const char* ${BIN2H_VARIABLE_NAME} = R\"delim(\n\n${asciiString}\n)delim\";\n")
