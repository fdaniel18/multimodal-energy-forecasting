from .database import engine, DATABASE_URL
from . import schema  # noqa: F401  (inregistreaza tabelele pe Base.metadata)
from .database import Base


def main():
    print(f"Conectare la: {DATABASE_URL}")
    Base.metadata.create_all(engine)
    print("Tabele create:", ", ".join(sorted(Base.metadata.tables.keys())))


if __name__ == "__main__":
    main()
