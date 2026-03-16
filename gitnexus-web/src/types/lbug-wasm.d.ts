declare module '@ladybugdb/wasm-core' {
  export function init(): Promise<void>;
  export class Database {
    constructor(path: string, bufferPoolSize?: number);
    close(): Promise<void>;
  }
  export class Connection {
    constructor(db: Database);
    query(cypher: string): Promise<QueryResult>;
    prepare(cypher: string): Promise<PreparedStatement>;
    execute(stmt: PreparedStatement, params?: Record<string, any>): Promise<QueryResult>;
    close(): Promise<void>;
  }
  export interface QueryResult {
    getAllRows(): Promise<any[]>;
    getAllObjects(): Promise<any[]>;
    getColumnNames(): Promise<string[]>;
    hasNext(): boolean;
    getNext(): Promise<any>;
    getNumTuples(): Promise<number>;
    close(): Promise<void>;
  }
  export interface PreparedStatement {
    isSuccess(): boolean;
    getErrorMessage(): Promise<string>;
    close(): Promise<void>;
  }
  export const FS: {
    writeFile(path: string, data: string): Promise<void>;
    unlink(path: string): Promise<void>;
  };
  const lbug: {
    init: typeof init;
    Database: typeof Database;
    Connection: typeof Connection;
    FS: typeof FS;
  };
  export default lbug;
}
