import psycopg2

def fetch_products_by_ids(search_ids, conn_string):
    """
    Fetches product details from the database for a given list of IDs.

    Args:
        search_ids: A list of product IDs to fetch.
        conn_string: database access details

    Returns:
        list: A list of tuples, where each tuple represents a product record. Returns an empty list if no products are found or if an error occurs.
    """
    if not search_ids:
        return []

    conn = None
    try:
        conn = psycopg2.connect(conn_string, sslmode = 'require')
        cur = conn.cursor()
        # Use the ANY operator for an efficient query with a list of IDs
        query = """
            SELECT id, title, price, image, description, miscellaneous
            FROM products
            WHERE id = ANY(%s)
            ORDER BY array_position(%s, id);
        """
        # psycopg2 requires a tuple of parameters. For a single parameter that is a list, it must be wrapped in a tuple.
        cur.execute(query, (search_ids, search_ids))
        results = cur.fetchall()
        return results

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return []
    except ValueError as e:
        print(f"Configuration error: {e}")
        return []
    finally:
        if conn:
            conn.close()